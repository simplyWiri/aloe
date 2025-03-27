#pragma once

#include <aloe/core/Device.h>

#include <tl/expected.hpp>

#include <unordered_set>

#include <slang-com-ptr.h>

namespace slang {
struct IGlobalSession;
struct IModule;
struct ISession;
struct PreprocessorMacroDesc;
}// namespace slang

namespace aloe {

struct ShaderInfo {
    std::string name;
    std::optional<std::string> shader_code = std::nullopt;
};

struct PipelineHandle {
    uint64_t id = 0;

    uint64_t operator()() const { return id; }
    auto operator<=>( const PipelineHandle& other ) const = default;
};

class PipelineManager {
    struct ConstructorArgs {
        std::string shader_path;
        Device& device;

        // Shader Settings
        std::vector<std::string> root_paths;
    };

    struct PipelineState {
        std::string path;
        uint64_t pipeline_id = 0;
        uint64_t version;

        std::vector<uint32_t> spirv;
    };

    Device& device_;
    std::vector<std::string> root_paths_;
    std::unordered_map<std::string, std::string> defines_;

    // Slang global session, for live compilation of shaders
    Slang::ComPtr<slang::IGlobalSession> global_session_ = nullptr;
    Slang::ComPtr<slang::ISession> session_ = nullptr;

    std::vector<PipelineState> pipelines_{};

public:
    PipelineManager( const ConstructorArgs& args );
    ~PipelineManager() = default;

    tl::expected<PipelineHandle, std::string> compile_pipeline( const ShaderInfo& shader_info,
                                                                std::string entry_point = "main" );
    void update_define( const std::string& name, const std::string& value );

    // Getters for pipelines
    uint64_t get_pipeline_version( PipelineHandle pid ) const { return pipelines_[pid.id].version; }
    const std::vector<uint32_t>& get_pipeline_spirv( PipelineHandle pid ) const { return pipelines_[pid.id].spirv; }

private:
    // Getters for internal state of the pipeline manager
    PipelineState& get_pipeline_state( const std::string& path );
    Slang::ComPtr<slang::ISession> get_session();

    // Shader processing
    tl::expected<Slang::ComPtr<slang::IModule>, std::string> compile_module( const ShaderInfo& shader_info );
    tl::expected<std::vector<uint32_t>, std::string> compile_spirv( Slang::ComPtr<slang::IModule> module,
                                                                    const std::string& entry_point_name );
};

}// namespace aloe