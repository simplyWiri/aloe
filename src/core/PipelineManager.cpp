#include <aloe/core/PipelineManager.h>
#include <aloe/util/log.h>

#include <slang.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <ranges>
#include <sstream>

namespace aloe {

struct SlangFilesystem : ISlangFileSystem {
    explicit SlangFilesystem( std::vector<std::string> root_paths ) : root_paths_( std::move( root_paths ) ) {}
    virtual ~SlangFilesystem() = default;

    // Add a file to the in-memory storage
    void set_file( std::string path, std::string content ) { files_[std::move( path )] = std::move( content ); }

    // Implementation of the loadFile method from ISlangFileSystem
    SlangResult loadFile( const char* path, ISlangBlob** outBlob ) override {
        // Slang adds a `-module` suffix to paths for modules; We need to get rid of this;
        std::string_view module_path = path;
        if ( module_path.ends_with( "-module" ) ) { module_path.remove_suffix( 7 ); }

        const auto it = files_.find( std::string( module_path ) );
        if ( it != files_.end() ) {
            // File found in memory; create a blob from the in-memory file content
            return create_blob( it->second, outBlob );
        }

        // File not found in memory; attempt to read from the standard file system
        for ( const auto& root_path : root_paths_ ) {
            const auto full_path = std::filesystem::canonical( std::filesystem::path{ root_path } / module_path );
            if ( std::filesystem::exists( full_path ) ) {
                std::ifstream file( full_path, std::ios::binary );

                // Read the file content into a vector
                const std::string buffer( ( std::istreambuf_iterator( file ) ), std::istreambuf_iterator<char>() );
                return create_blob( buffer, outBlob );
            }
        }

        return SLANG_E_NOT_FOUND;
    }

    uint32_t addRef() override { return 1; }
    uint32_t release() override { return 1; }
    SlangResult queryInterface( const SlangUUID&, void** ) override { return SLANG_E_NOT_IMPLEMENTED; }
    void* castAs( const SlangUUID& ) override { return nullptr; }

private:
    struct StringBlob : ISlangBlob {
        explicit StringBlob( std::string text ) : text_( std::move( text ) ), ref_count_( 1 ) {}
        virtual ~StringBlob() = default;

        SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface( const SlangUUID& guid, void** outObject ) override {
            if ( guid == ISlangBlob::getTypeGuid() || guid == ISlangUnknown::getTypeGuid() ) {
                *outObject = static_cast<ISlangBlob*>( this );
                addRef();
                return SLANG_OK;
            }
            *outObject = nullptr;
            return SLANG_E_NO_INTERFACE;
        }

        SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override { return ++ref_count_; }

        SLANG_NO_THROW uint32_t SLANG_MCALL release() override {
            const uint32_t count = --ref_count_;
            if ( count == 0 ) delete this;
            return count;
        }

        SLANG_NO_THROW const void* SLANG_MCALL getBufferPointer() override { return text_.data(); }
        SLANG_NO_THROW size_t SLANG_MCALL getBufferSize() override { return text_.size(); }

        std::string text_;
        std::atomic<uint32_t> ref_count_;
    };

    static SlangResult create_blob( std::string buffer, ISlangBlob** blob ) {
        *blob = new StringBlob( std::move( buffer ) );
        return SLANG_OK;
    }

    std::vector<std::string> root_paths_;
    std::unordered_map<std::string, std::string> files_;
};

PipelineManager::PipelineManager( Device& device, std::vector<std::string> root_paths )
    : device_( device )
    , root_paths_( std::move( root_paths ) ) {
    if ( SLANG_FAILED( slang::createGlobalSession( global_session_.writeRef() ) ) ) {
        throw std::runtime_error( "Failed to create Slang global session." );
    }

    filesystem_ = std::make_shared<SlangFilesystem>( root_paths_ );
}

tl::expected<PipelineHandle, std::string>
PipelineManager::compile_pipeline( const ComputePipelineInfo& compute_pipeline ) {
    auto module = compile_module( compute_pipeline.compute_shader );
    if ( !module ) { return tl::make_unexpected( module.error() ); }

    auto spirv = compile_spirv( *module, compute_pipeline.compute_shader.entry_point );
    if ( !spirv ) {
        return tl::make_unexpected(
            std::format( "{}: error: {}", compute_pipeline.compute_shader.name, spirv.error() ) );
    }

    auto& state = get_pipeline_state( compute_pipeline );
    state.version++;
    state.spirv = std::move( *spirv );
    return PipelineHandle{ state.id };
}

void PipelineManager::set_define( const std::string& name, const std::string& value ) {
    defines_[name] = value;
    session_ = nullptr;
}

void PipelineManager::set_virtual_file( const std::string& name, const std::string& contents ) {
    filesystem_->set_file( name, contents );
}

uint64_t PipelineManager::get_pipeline_version( PipelineHandle handle ) const {
    return pipelines_.at( handle.id ).version;
}

const std::vector<uint32_t>& PipelineManager::get_pipeline_spirv( PipelineHandle handle ) const {
    return pipelines_.at( handle.id ).spirv;
}

Slang::ComPtr<slang::ISession> PipelineManager::get_session() {
    // Rebuild the session if we have been asked, or it has not yet been set.
    if ( session_ == nullptr ) {
        const auto root_paths = root_paths_ | std::views::transform( []( const auto& path ) { return path.c_str(); } ) |
            std::ranges::to<std::vector>();
        const auto defines = defines_ | std::views::transform( []( const auto& pair ) -> slang::PreprocessorMacroDesc {
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
        session_desc.fileSystem = filesystem_.get();

        if ( SLANG_FAILED( global_session_->createSession( session_desc, session_.writeRef() ) ) ) {
            log_write( LogLevel::Error, "Failed to create Slang session." );
            return nullptr;
        }
    }

    return session_;
}

PipelineManager::PipelineState& PipelineManager::get_pipeline_state( const ComputePipelineInfo& pipeline_info ) {
    auto iter =
        std::ranges::find_if( pipelines_, [&]( const PipelineState& result ) { return result.info == pipeline_info; } );
    if ( iter != pipelines_.end() ) {
        return *iter;
    } else {
        const auto idx = pipelines_.size();
        pipelines_.emplace_back( PipelineState{ static_cast<uint32_t>( idx ), 0, pipeline_info, {} } );
        return pipelines_.back();
    }
}

tl::expected<Slang::ComPtr<slang::IModule>, std::string>
PipelineManager::compile_module( const ShaderCompileInfo& shader_info ) {
    const auto session = get_session();
    if ( session == nullptr ) { return tl::make_unexpected( "Failed to get session" ); }

    Slang::ComPtr<SlangCompileRequest> slang_request = nullptr;
    if ( SLANG_FAILED( session->createCompileRequest( slang_request.writeRef() ) ) ) {
        return tl::make_unexpected( "Failed to create compile request" );
    }

    // Compile our source file
    const auto tu_index = slang_request->addTranslationUnit( SLANG_SOURCE_LANGUAGE_SLANG, shader_info.name.c_str() );
    slang_request->addTranslationUnitSourceFile( tu_index, shader_info.name.c_str() );

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

    Slang::ComPtr<slang::IBlob> diagnostics;
    Slang::ComPtr<slang::IComponentType> linked_program = nullptr;
    if ( SLANG_FAILED( module->link( linked_program.writeRef(), diagnostics.writeRef() ) ) ||
         linked_program == nullptr ) {
        const auto str = std::string{ static_cast<const char*>( diagnostics->getBufferPointer() ) };
        return tl::make_unexpected( "Failed to link program" + str );
    }

    auto layout = linked_program->getLayout();
    auto entry_points = layout->getEntryPointCount();

    for ( uint32_t i = 0; i < entry_points; ++i ) {
        auto entry_point = layout->getEntryPointByIndex( i );
        std::cout << "Entry Point " << i << ": " << entry_point->getName() << std::endl;
    }

    return module;
}

tl::expected<std::vector<uint32_t>, std::string>
PipelineManager::compile_spirv( const Slang::ComPtr<slang::IModule>& module, const std::string& entry_point_name ) {
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
