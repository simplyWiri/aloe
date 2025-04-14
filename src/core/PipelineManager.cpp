#include <aloe/core/PipelineManager.h>
#include <aloe/core/aloe.slang>
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
    explicit SlangFilesystem( std::vector<std::string> root_paths ) : root_paths_( std::move( root_paths ) ) {
        // Hard-code the "root" shader which describes our I/O for the engine.
        files_["aloe.slang"] = SOURCE_STRING;
    }
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
    if ( pipeline != VK_NULL_HANDLE ) { vkDestroyPipeline( device.device(), pipeline, nullptr ); }
    if ( layout != VK_NULL_HANDLE ) { vkDestroyPipelineLayout( device.device(), layout, nullptr ); }

    for ( auto& shader : compiled_shaders ) {
        if ( shader.shader_module != VK_NULL_HANDLE ) {
            vkDestroyShaderModule( device.device(), shader.shader_module, nullptr );
        }
    }

    pipeline = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
    compiled_shaders.clear();
}

std::optional<std::string> PipelineManager::PipelineState::build_uniforms() {
    struct StageRange {
        VkShaderStageFlags stage;
        uint32_t begin;
        uint32_t end;
    };

    std::vector<StageRange> ranges;
    uint32_t global_max_offset = 0;

    for ( const auto& shader : compiled_shaders ) {
        if ( shader.uniforms.empty() ) continue;

        uint32_t min_offset = std::numeric_limits<uint32_t>::max();
        uint32_t max_offset = 0;

        for ( const auto& u : shader.uniforms ) {
            min_offset = std::min( min_offset, u.offset );
            max_offset = std::max( max_offset, u.offset + u.size );
        }

        ranges.emplace_back( shader.stage, min_offset, max_offset );
        global_max_offset = std::max( global_max_offset, max_offset );
    }

    // Check for overlaps between stage ranges
    for ( size_t i = 0; i < ranges.size(); ++i ) {
        for ( size_t j = i + 1; j < ranges.size(); ++j ) {
            const auto& a = ranges[i];
            const auto& b = ranges[j];

            if ( a.begin < b.end && b.begin < a.end ) {
                const auto a_name =
                    std::ranges::find_if( compiled_shaders, [&]( const auto& s ) { return s.stage == a.stage; } )->name;
                const auto b_name =
                    std::ranges::find_if( compiled_shaders, [&]( const auto& s ) { return s.stage == b.stage; } )->name;

                return std::format( "Overlapping stage push constant ranges:"
                                    "\n\t{} ({}) uses range [{}, {})"
                                    "\n\t{} ({}) uses range [{}, {})",
                                    a_name,
                                    a.stage,
                                    a.begin,
                                    a.end,
                                    b_name,
                                    b.stage,
                                    b.begin,
                                    b.end );
            }
        }
    }

    uniforms = UniformBlock{global_max_offset};

    return std::nullopt;
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

    create_global_descriptor_layout();
}

PipelineManager::~PipelineManager() {
    for ( auto& pipeline : pipelines_ ) { pipeline.free_state( device_ ); }

    if ( global_descriptor_set_ != VK_NULL_HANDLE ) {
        vkFreeDescriptorSets( device_.device(), global_descriptor_pool_, 1, &global_descriptor_set_ );
    }

    if ( global_descriptor_set_layout != VK_NULL_HANDLE ) {
        vkDestroyDescriptorSetLayout( device_.device(), global_descriptor_set_layout, nullptr );
    }

    if ( global_descriptor_pool_ != VK_NULL_HANDLE ) {
        vkDestroyDescriptorPool( device_.device(), global_descriptor_pool_, nullptr );
    }
}

tl::expected<PipelineHandle, std::string>
PipelineManager::compile_pipeline( const ComputePipelineInfo& compute_pipeline ) {
    auto& state = get_pipeline_state( compute_pipeline );
    state.free_state( device_ );// if we are re-compiling, ensure we free our prior (i.a) state

    const auto compiled_shader = get_compiled_shader( compute_pipeline.compute_shader );
    if ( !compiled_shader ) { return tl::make_unexpected( compiled_shader.error() ); }

    state.compiled_shaders = { *compiled_shader };

    if ( const auto uniform_error = state.build_uniforms() ) { return tl::make_unexpected( *uniform_error ); }

    const auto pipeline_layout = get_pipeline_layout( { *compiled_shader } );
    if ( !pipeline_layout ) { return tl::make_unexpected( pipeline_layout.error() ); }

    state.layout = *pipeline_layout;

    VkComputePipelineCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = compiled_shader->shader_module,
            .pName = compute_pipeline.compute_shader.entry_point.c_str(),
        },
        .layout = *pipeline_layout,
    };

    const auto result =
        vkCreateComputePipelines( device_.device(), VK_NULL_HANDLE, 1, &create_info, nullptr, &state.pipeline );
    if ( result != VK_SUCCESS ) {
        return tl::make_unexpected( std::format( "Failed to make compute pipeline, error: {}", result ) );
    }

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

        std::vector<slang::CompilerOptionEntry> options;
        options.emplace_back( slang::CompilerOptionEntry{ .name = slang::CompilerOptionName::VulkanUseEntryPointName,
                                                          .value = { slang::CompilerOptionValueKind::Int, 1 } } );

        auto session_desc = slang::SessionDesc{};
        session_desc.targets = &target_desc;
        session_desc.targetCount = 1;
        session_desc.searchPaths = root_paths.data();
        session_desc.searchPathCount = static_cast<SlangInt>( root_paths.size() );
        session_desc.preprocessorMacros = defines.data();
        session_desc.preprocessorMacroCount = static_cast<SlangInt>( defines.size() );
        session_desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;
        session_desc.fileSystem = filesystem_.get();
        session_desc.compilerOptionEntries = options.data();
        session_desc.compilerOptionEntryCount = options.size();

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
        pipelines_.emplace_back( PipelineState{
            .info = pipeline_info,
            .id = static_cast<uint32_t>( idx ),
            .version = 0,
        } );
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

void PipelineManager::create_global_descriptor_layout() {
    const auto limits = device_.get_physical_device_limits();

    // Make the descriptor pool.
    {

        std::vector<VkDescriptorPoolSize> pools;
        // Eventually we will extend this to support images etc.
        pools.emplace_back( VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, limits.maxDescriptorSetStorageBuffers );

        VkDescriptorPoolCreateInfo descriptor_pool_create_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags =
                VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
            .maxSets = 1,
            .poolSizeCount = static_cast<uint32_t>( pools.size() ),
            .pPoolSizes = pools.data(),
        };

        const auto result =
            vkCreateDescriptorPool( device_.device(), &descriptor_pool_create_info, nullptr, &global_descriptor_pool_ );
        if ( result != VK_SUCCESS ) { throw std::runtime_error{ "failed to create descriptor pool" }; }
    }

    // Make the descriptor set layout
    {
        std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
        std::vector<VkDescriptorBindingFlags> binding_flags;

        layout_bindings.emplace_back( ALOE_STORAGE_BUFFER_BINDING,
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      limits.maxDescriptorSetStorageBuffers,
                                      VK_SHADER_STAGE_ALL,
                                      nullptr );


        binding_flags.resize( layout_bindings.size() );
        std::ranges::fill( binding_flags,
                           VkDescriptorBindingFlags{ VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                                                     VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT } );

        VkDescriptorSetLayoutBindingFlagsCreateInfo layout_binding_flags_create_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            .bindingCount = static_cast<uint32_t>( binding_flags.size() ),
            .pBindingFlags = binding_flags.data(),
        };

        VkDescriptorSetLayoutCreateInfo set_layout_create_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = &layout_binding_flags_create_info,
            .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
            .bindingCount = static_cast<uint32_t>( layout_bindings.size() ),
            .pBindings = layout_bindings.data(),
        };

        const auto result = vkCreateDescriptorSetLayout( device_.device(),
                                                         &set_layout_create_info,
                                                         nullptr,
                                                         &global_descriptor_set_layout );
        if ( result != VK_SUCCESS ) { throw std::runtime_error{ "failed to create descriptor set layout" }; }
    }

    // Make the descriptor set
    {
        VkDescriptorSetAllocateInfo descriptor_set_allocate_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = global_descriptor_pool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &global_descriptor_set_layout,
        };

        const auto result =
            vkAllocateDescriptorSets( device_.device(), &descriptor_set_allocate_info, &this->global_descriptor_set_ );
        if ( result != VK_SUCCESS ) { throw std::runtime_error{ "failed to allocate descriptor set" }; }
    }
}

tl::expected<PipelineManager::CompiledShaderState, std::string>
PipelineManager::get_compiled_shader( const ShaderCompileInfo& shader_info ) {
    CompiledShaderState compiled_shader = {};

    if ( const auto module_error = compile_module( shader_info ) ) { return tl::make_unexpected( *module_error ); }
    if ( const auto spirv_error = compile_spirv( shader_info, compiled_shader.spirv ) ) {
        return tl::make_unexpected( *spirv_error );
    }

    // Populate our uniforms
    reflect_module( shader_info, compiled_shader );

    // Compile our `VkShaderModule`
    VkShaderModuleCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = compiled_shader.spirv.size() * sizeof( uint32_t ),
        .pCode = compiled_shader.spirv.data(),
    };

    const auto result = vkCreateShaderModule( device_.device(), &create_info, nullptr, &compiled_shader.shader_module );
    if ( result != VK_SUCCESS ) {
        return tl::make_unexpected( std::format( "Failed to create shader module, error: {}", result ) );
    }

    if ( device_.validation_enabled() ) {
        const auto name = std::format( "{}:{}", shader_info.name, shader_info.entry_point );

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

tl::expected<VkPipelineLayout, std::string>
PipelineManager::get_pipeline_layout( const std::vector<CompiledShaderState>& shaders ) {
    // Collect all push constant ranges from reflected uniforms
    std::vector<VkPushConstantRange> push_constants;
    for ( const auto& shader : shaders ) {
        // If we have any uniforms, we need to spec out a push constant
        if ( !shader.uniforms.empty() ) {
            assert( std::ranges::is_sorted( shader.uniforms ) );
            const auto last_pc = shader.uniforms.back();

            const auto stage = shader.stage;
            const auto end_offset = last_pc.offset + last_pc.size;

            // We can't register multiple push constants in Vulkan for the same stage, even if they are non-overlapping,
            // i.e. (range { offset: 0, size: 8 }, range { offset: 8, size: 4 } ) is an invaid construct, so we need to
            // merge each of the uniforms into a block of "contiguous" memory, as a singular push constant.
            push_constants.emplace_back( stage, 0, end_offset );
        }
    }

    VkPipelineLayoutCreateInfo pipeline_layout{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &global_descriptor_set_layout,
        .pushConstantRangeCount = static_cast<uint32_t>( push_constants.size() ),
        .pPushConstantRanges = push_constants.data(),
    };

    VkPipelineLayout layout = VK_NULL_HANDLE;
    const auto result = vkCreatePipelineLayout( device_.device(), &pipeline_layout, nullptr, &layout );
    if ( result != VK_SUCCESS ) {
        return tl::make_unexpected( std::format( "Failed to create pipeline layout, error: {}", result ) );
    }

    return layout;
}

}// namespace aloe
