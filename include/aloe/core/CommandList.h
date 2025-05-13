#pragma once

#include <aloe/core/PipelineManager.h>
#include <aloe/core/Resources.h>

#include <volk/volk.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "Resources.h"


namespace aloe {

class CommandList;

// Forward declarations
class PipelineManager;
class ResourceManager;

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

class BoundPipelineScope {
public:
    BoundPipelineScope( CommandList& cmd, PipelineManager& pipeline_manager, PipelineHandle handle );
    ~BoundPipelineScope() = default;

    // Set a uniform with its usage
    template<typename T>
    BoundPipelineScope& set_uniform( ShaderUniform<T> uniform, const T& value );
    // Compute dispatch
    void dispatch( uint32_t x, uint32_t y, uint32_t z );

private:
    void late_bind_buffer( BufferHandle handle );
    void late_bind_image( ImageHandle handle );

    CommandList& cmd_;
    PipelineManager& pipeline_manager_;
    PipelineHandle pipeline_;
    friend class CommandList;
};

class CommandList {
public:
    CommandList( const char* task_name,
                 VkCommandBuffer cmd,
                 PipelineManager& pipeline_mgr,
                 ResourceManager& resource_mgr );
    ~CommandList();

    // Pipeline binding returns RAII scope
    BoundPipelineScope bind_pipeline( PipelineHandle handle );

    // Renderpass management
    // void begin_renderpass( const RenderingInfo& info );
    // void end_renderpass();
private:
    VkCommandBuffer cmd_;
    PipelineManager& pipeline_mgr_;
    ResourceManager& resource_mgr_;

    friend class BoundPipelineScope;
};

}// namespace aloe
