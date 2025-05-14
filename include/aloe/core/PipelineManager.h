#pragma once

#include <volk.h>

#include <algorithm>
#include <expected>
#include <unordered_map>
#include <vector>

namespace aloe {

class Device;

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

    VkFormat color_attachment_format = VK_FORMAT_UNDEFINED;

    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPolygonMode polygon_mode = VK_POLYGON_MODE_FILL;
    VkCullModeFlags cull_mode = VK_CULL_MODE_BACK_BIT;
    VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    std::vector<VkDynamicState> dynamic_states = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    auto operator<=>( const GraphicsPipelineInfo& other ) const = default;
};

struct PipelineManager {
    PipelineManager( Device& device, std::vector<std::string> root_paths );
    ~PipelineManager();

    // Primary method for interaction with the API
    std::expected<PipelineHandle, std::string> compile_pipeline( const ComputePipelineInfo& pipeline_info );
    std::expected<PipelineHandle, std::string> compile_pipeline( const GraphicsPipelineInfo& pipeline_info );

    // Update the define(s) for all shaders being compiled
    void set_define( const std::string& name, const std::string& value );
    // Create a new virtual file which shaders can depend on
    void set_virtual_file( const std::string& path, const std::string& contents );

    template<typename T>
    ShaderUniform<T> get_uniform( PipelineHandle h, std::string_view name ) const;
};

}// namespace aloe