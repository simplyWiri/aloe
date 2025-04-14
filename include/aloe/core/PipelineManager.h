#pragma once

#include <aloe/core/Device.h>

#include <tl/expected.hpp>

#include <ranges>

#include <slang-com-ptr.h>

namespace slang {
struct IGlobalSession;
struct IModule;
struct ISession;
struct PreprocessorMacroDesc;
}// namespace slang

namespace aloe {

struct ShaderCompileInfo {
    std::string name;
    std::string entry_point = "main";

    auto operator<=>( const ShaderCompileInfo& other ) const = default;
};

struct ComputePipelineInfo {
    ShaderCompileInfo compute_shader = {};

    auto operator<=>( const ComputePipelineInfo& other ) const = default;
};

struct PipelineHandle {
    uint64_t id = 0;

    uint64_t operator()() const { return id; }
    auto operator<=>( const PipelineHandle& other ) const = default;
};

template<typename T>
struct ShaderUniform {
    static_assert( std::is_standard_layout_v<T> );
    explicit ShaderUniform( std::size_t offset ) : offset( offset ) {}

    std::size_t offset;
    std::optional<T> data;

    void set_value( const T& value ) { data = value; }
};

// We need to refactor the `UniformBlock` class to:
// 1. Maintain a vector<Stage, Offest, Size> for each shader it represents
// 2. Add verification that the total `data_` is <= device_.get_limits().maxPushConstantBytes;
// 3. Make the `bind` invocation, bind a push constant for each stage.

struct UniformBlock {
    explicit UniformBlock( std::size_t size = 0 ) { data_.resize( size ); }

    template<typename T>
    void set( const ShaderUniform<T>& element ) {
        assert( data_.size() >= element.offset + sizeof( T ) );
        std::memcpy( data_.data() + element.offset, &element.data.value(), sizeof( T ) );
    }

    void bind( VkCommandBuffer command_buffer, VkPipelineLayout layout ) const {
        vkCmdPushConstants( command_buffer, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, data_.size(), data_.data() );
    }

    auto operator<=>( const UniformBlock& other ) const = default;

    const void* data() const { return data_.data(); }
private:
    std::vector<uint8_t> data_;
};

// Pimpl style forward declaration for slang internals.
struct SlangFilesystem;

class PipelineManager {
    struct ShaderState {
        std::string name;

        /// Slang State Tracking
        Slang::ComPtr<SlangCompileRequest> compile_request = nullptr;
        Slang::ComPtr<slang::IModule> module = nullptr;

        /// Shader Dependency Tracking
        // Which other shader(s) does this shader depend on
        std::vector<ShaderState*> dependencies{};
        // Which other shader(s) rely on this shader
        std::vector<ShaderState*> dependents{};

        const std::vector<ShaderState*>& get_dependents() const;
    };

    struct CompiledShaderState {
        struct Uniform {
            uint32_t offset;
            uint32_t size;
            std::string name;

            auto operator<=>( const Uniform& other ) const = default;
        };

        std::string name;
        VkShaderStageFlags stage;
        VkShaderModule shader_module = VK_NULL_HANDLE;
        std::vector<uint32_t> spirv;
        std::vector<Uniform> uniforms;// sorted by `offset`

        auto operator<=>( const CompiledShaderState& other ) const = default;
    };

    struct PipelineState {
        ComputePipelineInfo info;
        uint32_t id;
        uint32_t version;

        // Vulkan Objects
        std::vector<CompiledShaderState> compiled_shaders;
        std::optional<UniformBlock> uniforms;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        void free_state( Device& device );
        std::optional<std::string> build_uniforms();
        bool matches_shader( const ShaderState& shader ) const;
        auto operator<=>( const PipelineState& other ) const = default;
    };

    Device& device_;
    std::vector<std::string> root_paths_;
    std::unordered_map<std::string, std::string> defines_;

    std::shared_ptr<SlangFilesystem> filesystem_ = nullptr;

    // Slang global session, for live compilation of shaders
    Slang::ComPtr<slang::IGlobalSession> global_session_ = nullptr;
    Slang::ComPtr<slang::ISession> session_ = nullptr;

    std::vector<PipelineState> pipelines_{};
    std::vector<std::unique_ptr<ShaderState>> shaders_{};

    VkDescriptorPool global_descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout global_descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorSet global_descriptor_set_ = VK_NULL_HANDLE;

public:
    PipelineManager( Device& device, std::vector<std::string> root_paths );
    ~PipelineManager();

    // Primary method for interaction with the API
    tl::expected<PipelineHandle, std::string> compile_pipeline( const ComputePipelineInfo& pipeline_info );

    // Update the define(s) for all shaders being compiled
    void set_define( const std::string& name, const std::string& value );
    // Create a new virtual file which shaders can depend on
    void set_virtual_file( const std::string& path, const std::string& contents );

    // Getters so unit tests can verify the validity of the code
    uint64_t get_pipeline_version( PipelineHandle h ) const;
    const std::vector<uint32_t>& get_pipeline_spirv( PipelineHandle h ) const;
    // const UniformBlock& get_uniform_block( PipelineHandle h, VkShaderStageFlags stage ) const;

    template<typename T>
    ShaderUniform<T> get_uniform_handle( PipelineHandle h, VkShaderStageFlags stage, std::string_view name ) const {
        for ( const auto& shader : pipelines_.at( h.id ).compiled_shaders ) {
            if ( shader.stage == stage ) {
                for ( const auto& uniform : shader.uniforms ) {
                    if ( uniform.name == name ) {
                        assert( uniform.size == sizeof( T ) );
                        return ShaderUniform<T>( uniform.offset );
                    }
                }
            }
        }

        return ShaderUniform<T>( 0 );
    }

private:
    // We need to rebuild our session when we change defines (as we ensure that all shaders are compiled with the same set of defines)
    Slang::ComPtr<slang::ISession> get_session();
    PipelineState& get_pipeline_state( const ComputePipelineInfo& pipeline_info );
    ShaderState& get_shader_state( const ShaderCompileInfo& path );

    // Shader processing, if string is returned - an error state has been set.
    std::optional<std::string> compile_module( const ShaderCompileInfo& info );
    std::optional<std::string> compile_spirv( const ShaderCompileInfo& info, std::vector<uint32_t>& spirv );
    // Populates `CompiledShaderState::uniforms`, i.e. push constants
    void reflect_module( const ShaderCompileInfo& info, CompiledShaderState& state );

    // Shader dependency tracking
    std::optional<std::string> update_shader_dependency_graph( const ShaderCompileInfo& info );
    void recompile_dependents( const std::vector<std::string>& shader_paths );

    // Vulkan object creation
    void create_global_descriptor_layout();
    tl::expected<CompiledShaderState, std::string> get_compiled_shader( const ShaderCompileInfo& info );
    tl::expected<VkPipelineLayout, std::string> get_pipeline_layout( const std::vector<CompiledShaderState>& shaders );
    tl::expected<UniformBlock, std::string> get_uniform_block( const std::vector<CompiledShaderState>& shaders);
};

}// namespace aloe