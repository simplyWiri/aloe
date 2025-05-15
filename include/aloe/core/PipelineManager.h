#pragma once

#include <aloe/core/Handles.h>

#include <volk.h>

#include <algorithm>
#include <expected>
#include <unordered_map>
#include <variant>
#include <vector>

#include "ResourceManager.h"
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

struct GraphicsPipelineInfo {
    ShaderCompileInfo vertex_shader = {};
    ShaderCompileInfo fragment_shader = {};

    auto operator<=>( const GraphicsPipelineInfo& other ) const = default;
};

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
            std::string type_name;

            auto operator<=>( const Uniform& other ) const = default;
        };

        std::string name;
        VkShaderStageFlags stage;
        VkShaderModule shader_module = VK_NULL_HANDLE;
        std::vector<uint32_t> spirv;
        std::vector<Uniform> uniforms;

        auto operator<=>( const CompiledShaderState& other ) const = default;
    };

    struct UniformBlock {
        explicit UniformBlock( uint32_t total_size ) { data_.resize( total_size, 0 ); }

        template<typename T>
        const T& get( const ShaderUniform<T>& element ) const {
            return *reinterpret_cast<const T*>( data_.data() + element.offset );
        }

        template<typename T>
        void set( const ShaderUniform<T>& element ) {
            assert( element.data.has_value() && "Attempting to set UniformBlock from ShaderUniform without data." );
            std::memcpy( data_.data() + element.offset, &element.data.value(), sizeof( T ) );
        }

        const void* data() const { return data_.data(); }
        std::size_t size() const { return data_.size(); }

        auto operator<=>( const UniformBlock& other ) const = default;

    private:
        std::vector<uint8_t> data_;
    };

    struct PipelineState {
        uint32_t id;
        uint32_t version;
        std::variant<GraphicsPipelineInfo, ComputePipelineInfo> info;

        std::vector<CompiledShaderState> compiled_shaders = {};

        std::optional<UniformBlock> uniforms = std::nullopt;
        std::vector<ResourceUsage> bound_resources = {};

        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        void free_state( Device& device );
        bool matches_shader( const ShaderState& shader ) const;
        void remove_resource( uint32_t resource_id );
        auto operator<=>( const PipelineState& other ) const = default;
    };

    Device& device_;
    ResourceManager& resource_manager_;
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
    std::expected<PipelineHandle, std::string> compile_pipeline( const GraphicsPipelineInfo& pipeline_info );

    // Update the define(s) for all shaders being compiled
    void set_define( const std::string& name, const std::string& value );
    // Create a new virtual file which shaders can depend on
    void set_virtual_file( const std::string& path, const std::string& contents );

    // Getters so unit tests can verify the validity of the code
    uint64_t get_pipeline_version( PipelineHandle ) const;
    const std::vector<uint32_t>& get_pipeline_spirv( PipelineHandle ) const;

    // todo: temporary API for binding a pipeline
    bool bind_pipeline( PipelineHandle handle, VkCommandBuffer buffer ) const;

    template<typename T>
        requires( std::is_standard_layout_v<T> )
    ShaderUniform<T> get_uniform_handle( PipelineHandle h, std::string_view name ) const {
        for ( const auto& shader : pipelines_.at( h.id ).compiled_shaders ) {
            for ( const auto& uniform : shader.uniforms ) {
                if ( uniform.name == name ) {
                    assert( uniform.size == sizeof( T ) );
                    return ShaderUniform<T>( h, uniform.offset );
                }
            }
        }
        assert( false );
        return ShaderUniform<T>( h, 0 );
    }

    template<typename T>
        requires( !(std::is_same_v<T, BufferHandle> || std::is_same_v<T, ImageHandle>) )
    void set_uniform( const ShaderUniform<T>& uniform ) {
        assert( uniform.data.has_value() && "Attempting to set UniformBlock from ShaderUniform without data." );
        pipelines_.at( uniform.pipeline.id ).uniforms->set( uniform );
    }

    template<typename T>
        requires( std::is_same_v<T, BufferHandle> || std::is_same_v<T, ImageHandle> )
    bool set_uniform( const ShaderUniform<T>& uniform, ResourceUsage usage ) {
        assert( uniform.data.has_value() && "Attempting to set UniformBlock from ShaderUniform without data." );
        assert( std::visit( [&]( auto& r ) { return r == *uniform.data; }, usage.resource ) );

        // Get the state we are referring too with `uniform::pipeline`
        auto& pipeline = pipelines_.at( uniform.pipeline.id );
        assert( pipeline.uniforms != std::nullopt );

        // If we have an old resource bound, we need to remove its reference
        const auto& old = pipeline.uniforms->get( uniform );
        pipeline.remove_resource( old >> 32 );

        // Try bind the resource (no-op if already bound)
        const auto slot = resource_manager_.bind_resource( usage );
        if ( slot == std::nullopt ) return false;

        // Write the encoded `slot+id` to the uniform block
        auto fake_uniform = ShaderUniform<T>( uniform.pipeline, uniform.offset );
        pipeline.uniforms->set( fake_uniform.set_value( *slot ) );
        pipeline.bound_resources.emplace_back( usage );

        return true;
    }

    void bind_slots() const;

private:
    // We need to rebuild our session when we change defines (as we ensure that all shaders are compiled with the same set of defines)
    template<typename PipelineInfoT>
    PipelineState& get_pipeline_state( const PipelineInfoT& pipeline_info ) {
        auto iter = std::ranges::find_if( pipelines_, [&]( const PipelineState& result ) {
            return std::holds_alternative<PipelineInfoT>( result.info ) &&
                std::get<PipelineInfoT>( result.info ) == pipeline_info;
        } );
        if ( iter != pipelines_.end() ) {
            return *iter;
        } else {
            const auto idx = pipelines_.size();
            pipelines_.emplace_back( PipelineState{ static_cast<uint32_t>( idx ), 0, pipeline_info, {} } );
            return pipelines_.back();
        }
    }

    Slang::ComPtr<slang::ISession> get_session();
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
    PipelineManager( Device& device, ResourceManager& resource_manager, std::vector<std::string> root_paths );
};

}// namespace aloe
