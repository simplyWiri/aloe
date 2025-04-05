#pragma once

#include <aloe/core/Device.h>

#include <tl/expected.hpp>

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

// Pimpl style forward declaration for slang internals.
struct SlangFilesystem;

class PipelineManager {
    // Used for tracking dependencies between shaders, eventually used for automatic re-compilation of pipelines
    struct ShaderState {
        std::string name;

        // Which other shader(s) does this shader depend on
        std::vector<ShaderState*> dependencies{};
        // Which other shader(s) rely on this shader
        std::vector<ShaderState*> dependents{};

        // If we have compiled this shader yet
        bool been_loaded = false;

        const std::vector<ShaderState*>& get_dependents() const;
    };

    struct PipelineState {
        uint32_t id;
        uint32_t version;
        ComputePipelineInfo info;

        // todo: Vulkan objects (VkPipeline)
        std::vector<uint32_t> spirv;

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

public:
    PipelineManager( Device& device, std::vector<std::string> root_paths );
    ~PipelineManager() = default;

    // Primary method for interaction with the API
    tl::expected<PipelineHandle, std::string> compile_pipeline( const ComputePipelineInfo& pipeline_info );

    // Update the define(s) for all shaders being compiled
    void set_define( const std::string& name, const std::string& value );
    // Create a new virtual file which shaders can depend on
    void set_virtual_file( const std::string& path, const std::string& contents );

    // Getters so unit tests can verify the validity of the code
    uint64_t get_pipeline_version( PipelineHandle ) const;
    const std::vector<uint32_t>& get_pipeline_spirv( PipelineHandle ) const;

private:
    // We need to rebuild our session when we change defines (as we ensure that all shaders are compiled with the same set of defines)
    Slang::ComPtr<slang::ISession> get_session();
    PipelineState& get_pipeline_state( const ComputePipelineInfo& pipeline_info );
    ShaderState& get_shader_state( const std::string& path );

    // Shader processing
    tl::expected<Slang::ComPtr<slang::IModule>, std::string> compile_module( const ShaderCompileInfo& shader_info );

    tl::expected<std::vector<uint32_t>, std::string> compile_spirv( const Slang::ComPtr<slang::IModule>& module,
                                                                    const std::string& entry_point_name );

    std::optional<std::string> update_shader_dependency_graph( const std::string& name,
                                                               const Slang::ComPtr<slang::IModule>& shader );

    void recompile_dependents( const std::vector<std::string>& shader_paths );
};

}// namespace aloe