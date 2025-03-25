#pragma once

#include <aloe/core/Device.h>

#include <tl/expected.hpp>

#include <slang-com-ptr.h>

namespace slang {
struct IGlobalSession;
}

namespace aloe {

class PipelineManager {
    struct ConstructorArgs {
        Device& device;

        // Shader Settings
        std::vector<std::string> root_paths;
        bool debug_info = false;
    };

    struct ShaderInfo {
        // Shader Name (filepath)
        std::string name;
        std::optional<std::string> shader_code = std::nullopt;
        std::string entry_point = "main";
        std::unordered_map<std::string, std::string> defines{};
    };

    Device& device_;
    std::vector<std::string> root_paths_;

    // Slang global session, for live compilation of shaders
    Slang::ComPtr<slang::IGlobalSession> global_session_;

public:
    PipelineManager( const ConstructorArgs& args );
    ~PipelineManager() = default;

    tl::expected<std::vector<uint32_t>, std::string> compile_shader( const ShaderInfo& shader_info );
};

}// namespace aloe