#pragma once

#include <aloe/core/Device.h>
#include <aloe/core/Handles.h>
#include <aloe/core/PipelineManager.h>

#include <volk.h>

#include <memory>
#include <optional>
#include <string_view>
#include <string>

namespace aloe {

class PipelineManager;
class ResourceManager;

struct SimulationState {
    // Monotonically increasing number each time `.execute()` is called on a task graph
    uint64_t sim_index;
    float time_since_epoch;
    // Time since the last tick as milliseconds, 0 for the first tick.
    float delta_time;
};


struct RenderingInfo {
    struct ColorAttachment {
        ImageHandle image;
        VkFormat format;
        VkAttachmentLoadOp load_op;
        VkAttachmentStoreOp store_op;
        VkClearValue clear_value;
    };

    std::vector<ColorAttachment> colors;
    std::optional<ColorAttachment> depth_stencil;
    VkRect2D render_area;
};

class CommandList;

class BoundPipelineScope {
public:
    template<typename T>
    BoundPipelineScope& set_uniform(const ShaderUniform<T>& uniform) {
        pipeline_manager_.set_uniform(uniform);
        return *this;
    }

    template<typename T>
    BoundPipelineScope& set_uniform(const ShaderUniform<T>& uniform, const ResourceUsage& usage) {
        pipeline_manager_.set_uniform(uniform, usage);
        return *this;
    }

    BoundPipelineScope& set_dynamic_state(VkDynamicState state, const void* data);

    std::optional<std::string> dispatch(uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z);
    std::optional<std::string> draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance);

private:
    friend class CommandList;
    BoundPipelineScope(CommandList& cmd_list, PipelineHandle handle, PipelineManager& pipeline_manager, bool in_renderpass);

    CommandList& cmd_list_;
    PipelineHandle pipeline_;
    PipelineManager& pipeline_manager_;
    bool is_graphics_pipeline_;
    bool is_in_renderpass_;
};

class CommandList {
public:
    CommandList(PipelineManager& pipeline_manager,
                ResourceManager& resource_manager,
                const char* section_name,
                VkCommandBuffer command_buffer,
                SimulationState simulation_state);
    ~CommandList();

    CommandList(const CommandList&) = delete;
    CommandList& operator=(const CommandList&) = delete;
    CommandList(CommandList&& other) = delete;
    CommandList& operator=(CommandList&& other) = delete;

    BoundPipelineScope bind_pipeline(PipelineHandle handle);

    std::optional<std::string> begin_renderpass(const RenderingInfo& info);
    std::optional<std::string> end_renderpass();
    bool in_renderpass() const { return in_renderpass_; }

    const SimulationState& state() const;

    void pipeline_barrier(const VkDependencyInfo& dependency_info ) const;

private:
    friend class BoundPipelineScope;

    PipelineManager& pipeline_manager_;
    ResourceManager& resource_manager_;
    SimulationState simulation_state_;
    VkCommandBuffer command_buffer_;
    bool in_renderpass_;
};

}// namespace aloe