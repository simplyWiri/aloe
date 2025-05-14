#pragma once

#include <aloe/core/handles.h>

#include <volk/volk.h>

#include <functional>
#include <vector>


namespace aloe {

struct SimulationState {
    // Monotonically increasing number each time `.execute()` is called on a task graph
    uint64_t sim_index;
    // Time since the last tick, 0 for the first tick.
    float delta_time;
};

struct RenderingInfo {
    struct ColorAttachment {
        ImageHandle image;
        VkFormat format;
        VkAttachmentLoadOp load_op = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        VkAttachmentStoreOp store_op = VK_ATTACHMENT_STORE_OP_STORE;
        VkClearValue clear_value = { .color = { 0.0f, 0.0f, 0.0f, 1.0f } };
    };

    std::vector<ColorAttachment> colors;
    std::optional<ColorAttachment> depth_stencil;
    VkRect2D render_area;
};

struct BoundPipelineScope {
    BoundPipelineScope( CommandList& cmd, PipelineManager& pipeline_manager, PipelineHandle handle );
    ~BoundPipelineScope() = default;

    template<typename T>
    BoundPipelineScope& set_uniform( ShaderUniform<T> uniform );
    BoundPipelineScope& set_dynamic_state( VkDynamicState state, const void* data );

    // Compute dispatch commands
    void dispatch( uint32_t x, uint32_t y, uint32_t z );

    // Graphics draw commands
    void
    draw( uint32_t vertex_count, uint32_t instance_count = 1, uint32_t first_vertex = 0, uint32_t first_instance = 0 );
    void draw_indexed( uint32_t index_count,
                       uint32_t instance_count = 1,
                       uint32_t first_index = 0,
                       int32_t vertex_offset = 0,
                       uint32_t first_instance = 0 );

private:
    CommandList& cmd;
    PipelineManager& pipeline_manager;
    PipelineHandle pipeline;
};

struct CommandList {
    CommandList( const char* task_name,
                 VkCommandBuffer cmd,
                 PipelineManager& pipeline_mgr,
                 ResourceManager& resource_mgr );
    ~CommandList();

    const SimulationState& state();

    BoundPipelineScope bind_pipeline( PipelineHandle handle );

    void begin_renderpass( const RenderingInfo& info );
    void end_renderpass();
};

}// namespace aloe
