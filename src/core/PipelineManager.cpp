#include <aloe/core/PipelineManager.h>
#include <aloe/util/algorithms.h>
#include <aloe/util/log.h>
#include <aloe/util/vulkan_util.h>

#include <slang.h>

#include <filesystem>
#include <fstream>
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
            const auto full_path = std::filesystem::path{ root_path } / module_path;
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

const std::vector<PipelineManager::ShaderState*>& PipelineManager::ShaderState::get_dependents() const {
    return dependents;
}

void PipelineManager::PipelineState::free_state( Device& device ) {
    for ( auto& shader : compiled_shaders ) {
        if ( shader.shader_module != VK_NULL_HANDLE ) {
            vkDestroyShaderModule( device.device(), shader.shader_module, nullptr );
        }
    }

    compiled_shaders.clear();
}

bool PipelineManager::PipelineState::matches_shader( const ShaderState& shader ) const {
    return shader.name == info.compute_shader.name;
}

PipelineManager::PipelineManager( Device& device, std::vector<std::string> root_paths )
    : device_( device )
    , root_paths_( std::move( root_paths ) ) {
    if ( SLANG_FAILED( slang::createGlobalSession( global_session_.writeRef() ) ) ) {
        throw std::runtime_error( "Failed to create Slang global session." );
    }

    filesystem_ = std::make_shared<SlangFilesystem>( root_paths_ );
}

PipelineManager::~PipelineManager() {
    for ( auto& pipeline : pipelines_ ) { pipeline.free_state( device_ ); }
}

std::expected<PipelineHandle, std::string>
PipelineManager::compile_pipeline( const ComputePipelineInfo& compute_pipeline ) {
    auto& state = get_pipeline_state( compute_pipeline );
    state.free_state( device_ );// if we are re-compiling, ensure we free our prior (i.a) state

    // Compile our shader for the pipeline
    const auto compiled_shader = get_compiled_shader( compute_pipeline.compute_shader );
    if ( !compiled_shader ) { return std::unexpected( compiled_shader.error() ); }
    state.compiled_shaders = { *compiled_shader };

    // Reflect and ensure that our uniform blocks (push constants) do not overlap.
    auto uniform_block = get_uniform_block( state.compiled_shaders );
    if ( !uniform_block ) { return std::unexpected( uniform_block.error() ); }
    state.uniforms = std::move( *uniform_block );

    state.version++;
    return PipelineHandle{ state.id };
}

void PipelineManager::set_define( const std::string& name, const std::string& value ) {
    defines_[name] = value;
    session_ = nullptr;

    // Recompile all shaders that have an entry point
    recompile_dependents( shaders_ | std::views::transform( []( const auto& shader ) { return shader->name; } ) |
                          std::ranges::to<std::vector>() );
}

void PipelineManager::set_virtual_file( const std::string& name, const std::string& contents ) {
    filesystem_->set_file( name, contents );

    recompile_dependents( { name } );
}

uint64_t PipelineManager::get_pipeline_version( PipelineHandle handle ) const {
    return pipelines_.at( handle.id ).version;
}

const std::vector<uint32_t>& PipelineManager::get_pipeline_spirv( PipelineHandle handle ) const {
    return pipelines_.at( handle.id ).compiled_shaders.front().spirv;
}

UniformBlock& PipelineManager::get_uniform_block( PipelineHandle handle ) {
    assert( pipelines_.at( handle.id ).uniforms );
    return *pipelines_.at( handle.id ).uniforms;
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

PipelineManager::ShaderState& PipelineManager::get_shader_state( const ShaderCompileInfo& info ) {
    auto iter =
        std::ranges::find_if( shaders_, [&]( const std::unique_ptr<ShaderState>& s ) { return s->name == info.name; } );
    if ( iter == shaders_.end() ) { return *shaders_.emplace_back( std::make_unique<ShaderState>( info.name ) ); }
    return **iter;
}

std::optional<std::string> PipelineManager::compile_module( const ShaderCompileInfo& info ) {
    auto& shader = get_shader_state( info );
    shader.module = nullptr;

    const auto session = get_session();
    if ( session == nullptr ) { return "Failed to get session"; }

    if ( SLANG_FAILED( session->createCompileRequest( shader.compile_request.writeRef() ) ) ) {
        return "Failed to create compile request";
    }

    // Compile our source file
    const auto tu_index = shader.compile_request->addTranslationUnit( SLANG_SOURCE_LANGUAGE_SLANG, info.name.c_str() );
    shader.compile_request->addTranslationUnitSourceFile( tu_index, info.name.c_str() );

    if ( SLANG_FAILED( shader.compile_request->compile() ) ) {
        const auto diagnostics = std::string{ shader.compile_request->getDiagnosticOutput() };
        return "Failed to compile shader (" + info.name + ": " + diagnostics;
    }

    // Compile the module for our shader
    if ( SLANG_FAILED( shader.compile_request->getModule( tu_index, shader.module.writeRef() ) ) ) {
        const auto diagnostics = std::string{ shader.compile_request->getDiagnosticOutput() };
        return "Failed to get module for compilation request (" + info.name + "), error: " + diagnostics;
    }

    // Update the shaders compilation information
    if ( const auto error = update_shader_dependency_graph( info ) ) {
        return "Failed to iterate dependencies, error: " + *error;
    }

    return std::nullopt;
}

std::optional<std::string> PipelineManager::compile_spirv( const ShaderCompileInfo& info,
                                                           std::vector<uint32_t>& spirv ) {
    const auto& shader = get_shader_state( info );
    assert( shader.module != nullptr );

    const auto session = get_session();
    if ( session == nullptr ) { return "Failed to get session"; }

    Slang::ComPtr<slang::IEntryPoint> entry_point = nullptr;
    if ( SLANG_FAILED( ( shader.module->findEntryPointByName( info.entry_point.c_str(), entry_point.writeRef() ) ) ) ) {
        return std::format( "Could not find entry point {}", info.entry_point );
    }

    Slang::ComPtr<slang::IBlob> diagnostics;
    Slang::ComPtr<slang::IComponentType> composite_program = nullptr;
    slang::IComponentType* components[] = { shader.module, entry_point.get() };
    if ( SLANG_FAILED( session->createCompositeComponentType( components,
                                                              2,
                                                              composite_program.writeRef(),
                                                              diagnostics.writeRef() ) ) ||
         composite_program == nullptr ) {
        const auto str = std::string{ static_cast<const char*>( diagnostics->getBufferPointer() ) };
        return "Failed to create composite program: " + str;
    }

    Slang::ComPtr<slang::IComponentType> linked_program = nullptr;
    if ( SLANG_FAILED( composite_program->link( linked_program.writeRef(), diagnostics.writeRef() ) ) ||
         linked_program == nullptr ) {
        const auto str = std::string{ static_cast<const char*>( diagnostics->getBufferPointer() ) };
        return "Failed to link program" + str;
    }

    Slang::ComPtr<slang::IBlob> code = nullptr;
    if ( SLANG_FAILED( linked_program->getEntryPointCode( 0, 0, code.writeRef(), diagnostics.writeRef() ) ) ) {
        const auto str = std::string{ static_cast<const char*>( diagnostics->getBufferPointer() ) };
        return "Failed to get entry point code blob, error: " + str;
    }

    // `getBufferSize()` is in bytes, we cast the `void*` to u32 (spirv operand size), so we divide by 4 when we
    // do our pointer arithmetic, to avoid over-reading.
    const auto* code_ptr = static_cast<const uint32_t*>( code->getBufferPointer() );
    spirv.assign( code_ptr, code_ptr + ( code->getBufferSize() / 4 ) );

    return std::nullopt;
}

std::optional<std::string> PipelineManager::update_shader_dependency_graph( const ShaderCompileInfo& info ) {
    auto& shader = get_shader_state( info );
    assert( shader.module != nullptr );

    // Fix any links pointing to our current shader as a dependent, as they may have been changed
    std::ranges::for_each( shader.dependencies, [&]( ShaderState* d ) { std::erase( d->dependents, &shader ); } );
    shader.dependencies.clear();

    for ( auto i = 0; i < shader.module->getDependencyFileCount(); ++i ) {
        auto file = std::string{ shader.module->getDependencyFilePath( i ) };
        if ( const auto dot_pos = file.find( '.' ); dot_pos != std::string_view::npos ) {
            if ( const auto colon_pos = file.find( ':', dot_pos ); colon_pos != std::string_view::npos ) {
                file.erase( colon_pos );
            }
        }

        auto& dependency_shader = get_shader_state( { .name = file } );
        if ( &dependency_shader == &shader ) { continue; }

        // Re-introduce our shader as a dependent of its dependencies
        shader.dependencies.emplace_back( &dependency_shader );
        dependency_shader.dependents.emplace_back( &shader );
    }

    return std::nullopt;
}

void PipelineManager::recompile_dependents( const std::vector<std::string>& shader_paths ) {
    const auto root_shaders = shader_paths |
        std::views::transform( [&]( const auto& path ) { return &get_shader_state( { .name = path } ); } );
    const auto all_shaders = aloe::topological_sort( root_shaders );
    auto all_pipelines = pipelines_ | std::views::filter( [&]( const auto& pipeline ) {
                             return std::ranges::any_of( all_shaders, [&]( const auto* shader ) {
                                 return pipeline.matches_shader( *shader );
                             } );
                         } );


    for ( const auto& pipeline : all_pipelines ) { compile_pipeline( pipeline.info ); }
}

void PipelineManager::reflect_module( const ShaderCompileInfo& info, CompiledShaderState& state ) {
    constexpr auto from_slang_stage = []( SlangStage stage ) -> VkShaderStageFlags {
        switch ( stage ) {
            case SlangStage::SLANG_STAGE_VERTEX: return VK_SHADER_STAGE_VERTEX_BIT;
            case SlangStage::SLANG_STAGE_FRAGMENT: return VK_SHADER_STAGE_FRAGMENT_BIT;
            case SlangStage::SLANG_STAGE_COMPUTE: return VK_SHADER_STAGE_COMPUTE_BIT;
            default:
                log_write( LogLevel::Error,
                           "Can not translate slang stage {} - returning `STAGE_ALL`",
                           static_cast<int>( stage ) );
        }

        return VK_SHADER_STAGE_ALL;
    };

    const auto& shader = get_shader_state( info );

    auto* reflection = reinterpret_cast<slang::ShaderReflection*>( shader.compile_request->getReflection() );
    auto* entry_point_reflection = reflection->findEntryPointByName( info.entry_point.c_str() );

    state.stage = from_slang_stage( entry_point_reflection->getStage() );

    for ( uint32_t i = 0; i < entry_point_reflection->getParameterCount(); ++i ) {
        const auto& param = entry_point_reflection->getParameterByIndex( i );

        if ( param->getCategory() == slang::Uniform ) {
            auto& uniform = state.uniforms.emplace_back();
            uniform.name = param->getName();
            uniform.offset = param->getOffset();
            uniform.size = param->getTypeLayout()->getSize();
        }
    }

    assert( std::ranges::is_sorted( state.uniforms ) );
}

std::expected<PipelineManager::CompiledShaderState, std::string>
PipelineManager::get_compiled_shader( const ShaderCompileInfo& info ) {
    CompiledShaderState compiled_shader = {};
    compiled_shader.name = info.name;

    if ( const auto module_error = compile_module( info ) ) { return std::unexpected( *module_error ); }
    if ( const auto spirv_error = compile_spirv( info, compiled_shader.spirv ) ) {
        return std::unexpected( *spirv_error );
    }

    // Populate our uniforms
    reflect_module( info, compiled_shader );

    // Compile our `VkShaderModule`
    VkShaderModuleCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = compiled_shader.spirv.size() * sizeof( uint32_t ),
        .pCode = compiled_shader.spirv.data(),
    };

    const auto result = vkCreateShaderModule( device_.device(), &create_info, nullptr, &compiled_shader.shader_module );
    if ( result != VK_SUCCESS ) {
        return std::unexpected( std::format( "Failed to create shader module, error: {}", result ) );
    }

    if ( device_.validation_enabled() ) {
        const auto name = std::format( "{}:{}", info.name, info.entry_point );

        VkDebugUtilsObjectNameInfoEXT debug_name_info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .objectType = VK_OBJECT_TYPE_SHADER_MODULE,
            .objectHandle = reinterpret_cast<uint64_t>( compiled_shader.shader_module ),
            .pObjectName = name.c_str(),
        };

        vkSetDebugUtilsObjectNameEXT( device_.device(), &debug_name_info );
    }

    return compiled_shader;
}

std::expected<UniformBlock, std::string>
PipelineManager::get_uniform_block( const std::vector<CompiledShaderState>& shaders ) {
    std::vector<UniformBlock::StageBinding> stage_bindings;
    uint32_t global_max_offset = 0;// Tracks the highest byte offset needed across all stages

    // track ranges per stage for overlap check (stage, min_offset, max_end_offset)
    std::vector<std::tuple<VkShaderStageFlags, uint32_t, uint32_t>> stage_ranges;

    for ( const auto& shader : shaders ) {
        if ( shader.uniforms.empty() ) continue;

        // Ensure uniforms are sorted by offset (should be guaranteed by reflect_module)
        assert( std::ranges::is_sorted( shader.uniforms, {}, &CompiledShaderState::Uniform::offset ) );

        const uint32_t stage_min_offset = shader.uniforms.front().offset;
        const uint32_t stage_max_end_offset = shader.uniforms.back().offset + shader.uniforms.back().size;
        const uint32_t stage_size = stage_max_end_offset - stage_min_offset;

        assert( stage_size > 0 );

        // Update the overall maximum offset needed for the whole block
        global_max_offset = std::max( global_max_offset, stage_max_end_offset );

        stage_bindings.emplace_back( shader.stage, stage_min_offset, stage_size );
        stage_ranges.emplace_back( shader.stage, stage_min_offset, stage_max_end_offset );
    }

    for ( size_t i = 0; i + 1 < stage_ranges.size(); ++i ) {
        const auto& [cur_stage, cur_min, cur_max] = stage_ranges[i];
        const auto& [nxt_stage, nxt_min, nxt_max] = stage_ranges[i + 1];

        // Check if current range's end overlaps next range's start
        if ( cur_max > nxt_min ) {
            // Find shader names for better error message
            auto cur_shader = std::ranges::find_if( shaders, [&]( const auto& s ) { return s.stage == cur_stage; } );
            auto nxt_shader = std::ranges::find_if( shaders, [&]( const auto& s ) { return s.stage == nxt_stage; } );

            return std::unexpected( std::format( "Overlapping push constant data ranges detected between stages:\n"
                                                 "  Shader '{}' (Stage {}) uses range [{}, {})\n"
                                                 "  Shader '{}' (Stage {}) uses range [{}, {})",
                                                 cur_shader->name,
                                                 cur_stage,
                                                 cur_min,
                                                 cur_max,
                                                 nxt_shader->name,
                                                 nxt_stage,
                                                 nxt_min,
                                                 nxt_max ) );
        }
    }

    // Check if we are trying to create a push constant greater than our physical device capacities
    if ( global_max_offset > device_.get_physical_device_limits().maxPushConstantsSize ) {
        return std::unexpected(
            std::format( "Total push constant size required ({}) exceeds device limit (maxPushConstantsSize = {}).",
                         global_max_offset,
                         device_.get_physical_device_limits().maxPushConstantsSize ) );
    }

    return UniformBlock( stage_bindings, global_max_offset );
}

}// namespace aloe
