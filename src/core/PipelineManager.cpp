#include <aloe/core/PipelineManager.h>

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


tl::expected<std::vector<uint32_t>, std::string> PipelineManager::compile_shader( const ShaderInfo& shader_info ) {
    auto session = Slang::ComPtr<slang::ISession>{};

    // Create
    {
        const auto search_paths = root_paths_ | std::views::transform( []( const auto& x ) { return x.c_str(); } ) |
            std::ranges::to<std::vector>();
        const auto defines = shader_info.defines |
            std::views::transform( []( const auto& x ) -> slang::PreprocessorMacroDesc {
                                 return { x.first.c_str(), x.second.c_str() };
                             } ) |
            std::ranges::to<std::vector>();

        auto target_desc = slang::TargetDesc{};
        target_desc.format = SlangCompileTarget::SLANG_SPIRV;
        target_desc.profile = global_session_->findProfile( "spirv_1_5" );
        target_desc.flags = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;

        auto session_desc = slang::SessionDesc{};
        session_desc.targets = &target_desc;
        session_desc.targetCount = 1;
        session_desc.searchPaths = search_paths.data();
        session_desc.searchPathCount = search_paths.size();
        session_desc.preprocessorMacros = defines.data();
        session_desc.preprocessorMacroCount = defines.size();
        session_desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;

        if ( SLANG_FAILED( global_session_->createSession( session_desc, session.writeRef() ) ) ) {
            return tl::make_unexpected( "Failed to create session, no error was provided" );
        }
    }

    Slang::ComPtr<SlangCompileRequest> slang_request = nullptr;
    session->createCompileRequest( slang_request.writeRef() );

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

    Slang::ComPtr<slang::IEntryPoint> entry_point = nullptr;
    if ( SLANG_FAILED( module->findEntryPointByName( shader_info.entry_point.c_str(), entry_point.writeRef() ) ) ) {
        return tl::make_unexpected( "Could not find entry point " + shader_info.entry_point + " in shader " +
                                    shader_info.name );
    }

    Slang::ComPtr<slang::IBlob> diagnostics;
    Slang::ComPtr<slang::IComponentType> composite_program = nullptr;
    slang::IComponentType* components[] = { module, entry_point.get() };
    if ( SLANG_FAILED( session->createCompositeComponentType( components,
                                                              2,
                                                              composite_program.writeRef(),
                                                              diagnostics.writeRef() ) ) ||
         composite_program == nullptr ) {
        return tl::unexpected( "Failed to create composite program for " + shader_info.name );
    }

    Slang::ComPtr<slang::IComponentType> linked_program = nullptr;
    if ( SLANG_FAILED( composite_program->link( linked_program.writeRef(), diagnostics.writeRef() ) ) ||
         linked_program == nullptr ) {
        return tl::unexpected( "Failed to link program for " + shader_info.name );
    }

    Slang::ComPtr<slang::IBlob> code = nullptr;
    if ( SLANG_FAILED( linked_program->getEntryPointCode( 0, 0, code.writeRef(), diagnostics.writeRef() ) ) ) {
        const auto str = std::string{ static_cast<const char*>( diagnostics->getBufferPointer() ) };
        return tl::make_unexpected( "Failed to get entry point code blob: " + str );
    }

    // `getBufferSize()` is in bytes, we cast the `void*` to u32 (spirv operand size), so we divide by 4 when we
    // do our pointer arithmetic, to avoid over-reading.
    const auto* code_ptr = static_cast<const uint32_t*>( code->getBufferPointer() );
    return std::vector( code_ptr, code_ptr + ( code->getBufferSize() / 4 ) );
}

}// namespace aloe
