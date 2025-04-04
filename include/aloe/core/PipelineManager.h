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

struct ShaderCompileInfo {
    std::optional<std::string> path;
    std::optional<std::string> source;
    std::optional<std::string> entry_point;
};

struct ComputePipelineInfo {
    ShaderCompileInfo compute_shader = {};
};

struct PipelineHandle {
    uint64_t id = 0;

    uint64_t operator()() const { return id; }
    auto operator<=>( const PipelineHandle& other ) const = default;
};

class PipelineManager {
    Device& device_;
    std::vector<std::string> root_paths_;
    std::unordered_map<std::string, std::string> defines_;

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
};

}// namespace aloe