#include <aloe/core/PipelineManager.h>
#include <aloe/util/log.h>

#include <slang.h>

#include <filesystem>
#include <iostream>
#include <ranges>

namespace aloe {

PipelineManager::PipelineManager( const ConstructorArgs& args )
    : device_( args.device )
    , root_paths_( args.root_paths ) {
    if ( SLANG_FAILED( slang::createGlobalSession( global_session_.writeRef() ) ) ) {
        throw std::runtime_error( "Failed to create Slang global session." );
    }
}

void PipelineManager::update_define( const std::string& name, const std::string& value ) {
    if ( const auto it = defines_.find( name ); it != defines_.end() ) {
        it->second = value;
    } else {
        defines_.emplace( name, value );
    }

    // Rebuild the session on next lookup, a global define has changed.
    session_ = nullptr;
}

tl::expected<PipelineHandle, std::string> PipelineManager::compile_pipeline( const ShaderInfo& shader_info,
                                                                             std::string entry_point_name ) {
    auto module = compile_module( shader_info );
    if ( !module ) { return tl::make_unexpected( module.error() ); }

    auto spirv = compile_spirv( *module, entry_point_name );
    if ( !spirv ) { return tl::make_unexpected( std::format( "{}: error: {}", shader_info.name, spirv.error() ) ); }

    const auto shader_path = ( *module )->getFilePath();

    auto& state = get_pipeline_state( shader_path );
    state.version++;
    state.spirv = std::move( *spirv );
    return PipelineHandle{ state.pipeline_id };
}

PipelineManager::PipelineState& PipelineManager::get_pipeline_state( const std::string& path ) {
    auto iter = std::ranges::find_if( pipelines_, [&]( const PipelineState& result ) { return result.path == path; } );
    if ( iter != pipelines_.end() ) {
        return *iter;
    } else {
        const auto idx = pipelines_.size();
        pipelines_.emplace_back( PipelineState{ path, idx, 0, {} } );
        return pipelines_.back();
    }
}

Slang::ComPtr<slang::ISession> PipelineManager::get_session() {
    // Rebuild the session if we have been asked, or it has not (yet) been set.
    if ( session_ == nullptr ) {
        auto root_paths = root_paths_ | std::views::transform( []( const auto& path ) { return path.c_str(); } ) |
            std::ranges::to<std::vector>();
        auto defines = defines_ | std::views::transform( []( const auto& pair ) -> slang::PreprocessorMacroDesc {
                           return { pair.first.c_str(), pair.second.c_str() };
                       } ) |
            std::ranges::to<std::vector>();

        auto target_desc = slang::TargetDesc{};
        target_desc.format = SlangCompileTarget::SLANG_SPIRV;
        target_desc.profile = global_session_->findProfile( "spirv_1_5" );
        target_desc.flags = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;

        auto session_desc = slang::SessionDesc{};
        session_desc.targets = &target_desc;
        session_desc.targetCount = 1;
        session_desc.searchPaths = root_paths.data();
        session_desc.searchPathCount = static_cast<SlangInt>( root_paths.size() );
        session_desc.preprocessorMacros = defines.data();
        session_desc.preprocessorMacroCount = static_cast<SlangInt>( defines.size() );
        session_desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;

        if ( SLANG_FAILED( global_session_->createSession( session_desc, session_.writeRef() ) ) ) {
            log_write( LogLevel::Error, "Failed to create Slang session." );
            return nullptr;
        }
    }

    return session_;
}

tl::expected<Slang::ComPtr<slang::IModule>, std::string>
PipelineManager::compile_module( const ShaderInfo& shader_info ) {
    const auto session = get_session();
    if ( session == nullptr ) { return tl::make_unexpected( "Failed to get session" ); }

    Slang::ComPtr<SlangCompileRequest> slang_request = nullptr;
    if ( SLANG_FAILED( session->createCompileRequest( slang_request.writeRef() ) ) ) {
        return tl::make_unexpected( "Failed to create compile request" );
    }

    // Compile our source file
    const auto tu_index = slang_request->addTranslationUnit( SLANG_SOURCE_LANGUAGE_SLANG, shader_info.name.c_str() );

    // if `shader_code` == std::nullopt, we want to load the file from disk, otherwise we can load it from memory.
    if ( shader_info.shader_code ) {
        slang_request->addTranslationUnitSourceString( tu_index,
                                                       shader_info.name.c_str(),
                                                       shader_info.shader_code->c_str() );
    } else {
        for ( const auto& root_path : root_paths_ ) {
            const auto resolved_path = std::filesystem::path( root_path ) / shader_info.name;

            if ( std::filesystem::exists( resolved_path ) ) {
                slang_request->addTranslationUnitSourceFile( tu_index, resolved_path.c_str() );
            }
        }
    }

    if ( SLANG_FAILED( slang_request->compile() ) ) {
        const auto diagnostics = std::string{ slang_request->getDiagnosticOutput() };
        return tl::make_unexpected( "Failed to compile shader (" + shader_info.name + ": " + diagnostics );
    }

    Slang::ComPtr<slang::IModule> module = nullptr;
    if ( SLANG_FAILED( slang_request->getModule( tu_index, module.writeRef() ) ) ) {
        const auto diagnostics = std::string{ slang_request->getDiagnosticOutput() };
        return tl::make_unexpected( "Failed to get module for compilation request (" + shader_info.name +
                                    "), error: " + diagnostics );
    }

    return module;
}

tl::expected<std::vector<uint32_t>, std::string> PipelineManager::compile_spirv( Slang::ComPtr<slang::IModule> module,
                                                                                 const std::string& entry_point_name ) {
    const auto session = get_session();
    if ( session == nullptr ) { return tl::make_unexpected( "Failed to get session" ); }

    Slang::ComPtr<slang::IEntryPoint> entry_point = nullptr;

    if ( SLANG_FAILED( ( module->findEntryPointByName( entry_point_name.c_str(), entry_point.writeRef() ) ) ) ) {
        return tl::make_unexpected( std::format( "Could not find entry point {}", entry_point_name ) );
    }

    Slang::ComPtr<slang::IBlob> diagnostics;
    Slang::ComPtr<slang::IComponentType> composite_program = nullptr;
    slang::IComponentType* components[] = { module, entry_point.get() };
    if ( SLANG_FAILED( session->createCompositeComponentType( components,
                                                              2,
                                                              composite_program.writeRef(),
                                                              diagnostics.writeRef() ) ) ||
         composite_program == nullptr ) {
        const auto str = std::string{ static_cast<const char*>( diagnostics->getBufferPointer() ) };
        return tl::make_unexpected( "Failed to create composite program: " + str );
    }

    Slang::ComPtr<slang::IComponentType> linked_program = nullptr;
    if ( SLANG_FAILED( composite_program->link( linked_program.writeRef(), diagnostics.writeRef() ) ) ||
         linked_program == nullptr ) {
        const auto str = std::string{ static_cast<const char*>( diagnostics->getBufferPointer() ) };
        return tl::make_unexpected( "Failed to link program" + str );
    }

    Slang::ComPtr<slang::IBlob> code = nullptr;
    if ( SLANG_FAILED( linked_program->getEntryPointCode( 0, 0, code.writeRef(), diagnostics.writeRef() ) ) ) {
        const auto str = std::string{ static_cast<const char*>( diagnostics->getBufferPointer() ) };
        return tl::make_unexpected( "Failed to get entry point code blob, error: " + str );
    }

    // `getBufferSize()` is in bytes, we cast the `void*` to u32 (spirv operand size), so we divide by 4 when we
    // do our pointer arithmetic, to avoid over-reading.
    const auto* code_ptr = static_cast<const uint32_t*>( code->getBufferPointer() );
    return std::vector( code_ptr, code_ptr + ( code->getBufferSize() / 4 ) );
}

}// namespace aloe
