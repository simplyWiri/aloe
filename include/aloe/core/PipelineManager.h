#pragma once

#include <aloe/core/Resources.h>

#include <volk.h>

#include <algorithm>
#include <expected>
#include <unordered_map>
#include <vector>

#include <slang-com-ptr.h>

namespace slang {
struct IGlobalSession;
struct IModule;
struct ISession;
struct PreprocessorMacroDesc;
}// namespace slang

namespace aloe {
class Device;
class ResourceManager;

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
    explicit ShaderUniform( uint32_t offset ) : offset( offset ) {}

    uint32_t offset;// Using uint32_t
    std::optional<T> data;

    ShaderUniform set_value( const T& value ) {
        data = value;
        return *this;
    }
    auto operator<=>( const ShaderUniform& ) const = default;
};

class UniformBlock {
public:
    struct StageBinding {
        VkShaderStageFlags stage_flags = 0;
        uint32_t offset = 0;
        uint32_t size = 0;
        auto operator<=>( const StageBinding& ) const = default;
    };

    explicit UniformBlock( const std::vector<StageBinding>& bindings, uint32_t total_size )
        : stage_bindings_( bindings ) {
        data_.resize( total_size );
        std::ranges::fill( data_, uint8_t{ 0 } );
    }

    template<typename T>
    void set( const ShaderUniform<T>& element ) {
        assert( element.data.has_value() && "Attempting to set UniformBlock from ShaderUniform without data." );
        std::memcpy( data_.data() + element.offset, &element.data.value(), sizeof( T ) );
    }

    const void* data() const { return data_.data(); }
    std::size_t size() const { return data_.size(); }
    const std::vector<StageBinding>& get_bindings() const { return stage_bindings_; }

    auto operator<=>( const UniformBlock& other ) const = default;

private:
    std::vector<uint8_t> data_;
    std::vector<StageBinding> stage_bindings_;
};

// Pimpl style forward declaration for slang internals.
struct SlangFilesystem;

class PipelineManager {
    friend class Device;
    // Represents a shader file on disk, that has been linked to its dependencies, but has not yet been
    // compiled for a particular entry point, you need an `entry_point` name to turn this into a
    // `CompiledShaderState` object.
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

    // Represents a shader that has been compiled for a given entry point (& stage)
    struct CompiledShaderState {
        struct Uniform {
            uint32_t offset;
            uint32_t size;
            std::string name;

            auto operator<=>( const Uniform& other ) const = default;
        };

        std::string name;// maps to `ShaderState::name`
        VkShaderStageFlags stage;
        VkShaderModule shader_module = VK_NULL_HANDLE;
        std::vector<uint32_t> spirv;  // todo: not necessary to store, stored for tests.
        std::vector<Uniform> uniforms;// sorted by `offset`

        auto operator<=>( const CompiledShaderState& other ) const = default;
    };

    struct PipelineState {
        uint32_t id;
        uint32_t version;
        ComputePipelineInfo info;

        std::vector<CompiledShaderState> compiled_shaders = {};
        std::optional<UniformBlock> uniforms = std::nullopt;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        void free_state( Device& device );
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
    ~PipelineManager();

    PipelineManager( PipelineManager& ) = delete;
    PipelineManager& operator=( const PipelineManager& other ) = delete;

    PipelineManager( PipelineManager&& ) = delete;
    PipelineManager& operator=( PipelineManager&& other ) = delete;

    // Primary method for interaction with the API
    std::expected<PipelineHandle, std::string> compile_pipeline( const ComputePipelineInfo& pipeline_info );

    // Update the define(s) for all shaders being compiled
    void set_define( const std::string& name, const std::string& value );
    // Create a new virtual file which shaders can depend on
    void set_virtual_file( const std::string& path, const std::string& contents );

    // Getters so unit tests can verify the validity of the code
    uint64_t get_pipeline_version( PipelineHandle ) const;
    const std::vector<uint32_t>& get_pipeline_spirv( PipelineHandle ) const;

    // todo: temporary API for binding a resource
    VkResult bind_buffer( ResourceManager& resource_manager,
                          BufferHandle buffer_handle,
                          VkDeviceSize offset = 0,
                          VkDeviceSize range = VK_WHOLE_SIZE );

    // todo: temporary APIs before we lift this into a higher level (CommandList) type API.
    UniformBlock& get_uniform_block( PipelineHandle h );
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
        assert( false );
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
    std::optional<std::string> update_shader_dependency_graph( const ShaderCompileInfo& info );

    void recompile_dependents( const std::vector<std::string>& shader_paths );
    void reflect_module( const ShaderCompileInfo& info, CompiledShaderState& state );

    void create_global_descriptor_layout();
    std::expected<CompiledShaderState, std::string> get_compiled_shader( const ShaderCompileInfo& info );
    std::expected<UniformBlock, std::string> get_uniform_block( const std::vector<CompiledShaderState>& shaders );
    std::expected<VkPipelineLayout, std::string> get_pipeline_layout( const std::vector<CompiledShaderState>& shaders );

protected:
    PipelineManager( Device& device, std::vector<std::string> root_paths );
};

}// namespace aloe