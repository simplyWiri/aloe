#include <aloe/core/CommandList.h>
#include <aloe/core/PipelineManager.h>
#include <aloe/core/ResourceManager.h>
#include <aloe/util/log.h>

#include <cassert>
#include <optional>
#include <string>

namespace aloe {

BoundPipelineScope::BoundPipelineScope( CommandList& cmd_list,
                                        PipelineHandle handle,
                                        PipelineManager& pipeline_manager,
                                        bool in_renderpass )
    : cmd_list_( cmd_list )
    , pipeline_( handle )
    , pipeline_manager_( pipeline_manager )
    , is_graphics_pipeline_( pipeline_manager.is_graphics_pipeline( handle ) )
    , is_in_renderpass_( in_renderpass ) {
}

BoundPipelineScope& BoundPipelineScope::set_dynamic_state( VkDynamicState state, const void* data ) {
    switch ( state ) {
        case VK_DYNAMIC_STATE_VIEWPORT: {
            const auto* viewport = static_cast<const VkViewport*>( data );
            vkCmdSetViewport( cmd_list_.command_buffer_, 0, 1, viewport );
            break;
        }
        case VK_DYNAMIC_STATE_SCISSOR: {
            const auto* scissor = static_cast<const VkRect2D*>( data );
            vkCmdSetScissor( cmd_list_.command_buffer_, 0, 1, scissor );
            break;
        }
        // Add other dynamic states as needed
        default: assert( false && "Unsupported dynamic state" ); break;
    }
    return *this;
}

std::optional<std::string>
BoundPipelineScope::dispatch( uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z ) {
    if ( is_graphics_pipeline_ ) { return "Cannot dispatch with a graphics pipeline"; }
    if ( is_in_renderpass_ ) { return "Cannot dispatch inside a render pass"; }

    if ( !pipeline_manager_.bind_pipeline( pipeline_, cmd_list_.command_buffer_ ) ) {
        return "Failed to bind compute pipeline";
    }

    vkCmdDispatch( cmd_list_.command_buffer_, group_count_x, group_count_y, group_count_z );
    return std::nullopt;
}

std::optional<std::string> BoundPipelineScope::draw( uint32_t vertex_count,
                                                     uint32_t instance_count,
                                                     uint32_t first_vertex,
                                                     uint32_t first_instance ) {
    if ( !is_graphics_pipeline_ ) { return "Cannot draw with a compute pipeline"; }
    if ( !is_in_renderpass_ ) { return "Cannot draw outside of a render pass"; }

    if ( !pipeline_manager_.bind_pipeline( pipeline_, cmd_list_.command_buffer_ ) ) {
        return "Failed to bind graphics pipeline";
    }

    vkCmdDraw( cmd_list_.command_buffer_, vertex_count, instance_count, first_vertex, first_instance );
    return std::nullopt;
}

CommandList::CommandList( PipelineManager& pipeline_manager,
                          ResourceManager& resource_manager,
                          const char* section_name,
                          VkCommandBuffer command_buffer,
                          SimulationState simulation_state )
    : pipeline_manager_( pipeline_manager )
    , resource_manager_( resource_manager )
    , simulation_state_( simulation_state )
    , command_buffer_( command_buffer )
    , in_renderpass_( false ) {

    const VkDebugUtilsLabelEXT label_info{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
        .pLabelName = section_name,
        .color = { 1.0f, 1.0f, 1.0f, 1.0f },
    };
    vkCmdBeginDebugUtilsLabelEXT( command_buffer_, &label_info );
}

CommandList::~CommandList() {
    if ( in_renderpass_ ) {
        log_write( LogLevel::Error, "Renderpass was not ended before CommandList destruction" );
        vkCmdEndRenderingKHR( command_buffer_ );
    }
    vkCmdEndDebugUtilsLabelEXT( command_buffer_ );
}

const SimulationState& CommandList::state() const {
    return simulation_state_;
}

BoundPipelineScope CommandList::bind_pipeline( PipelineHandle handle ) {
    return { *this, handle, pipeline_manager_, in_renderpass_ };
}

std::optional<std::string> CommandList::begin_renderpass( const RenderingInfo& info ) {
    if ( in_renderpass_ ) { return "Already in render pass"; }

    std::vector<VkRenderingAttachmentInfoKHR> color_attachments;
    color_attachments.reserve( info.colors.size() );

    for ( const auto& color : info.colors ) {
        const auto colour_usage = usage( color.image, ColorAttachmentWrite );
        color_attachments.emplace_back( VkRenderingAttachmentInfoKHR{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
            .imageView = resource_manager_.get_image_view( colour_usage ),
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = color.load_op,
            .storeOp = color.store_op,
            .clearValue = color.clear_value,
        } );
    }

    VkRenderingAttachmentInfoKHR depth_attachment{};
    if ( info.depth_stencil ) {
        const auto depth_usage = usage( info.depth_stencil->image, DepthStencilAttachmentWrite );
        depth_attachment = VkRenderingAttachmentInfoKHR{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
            .imageView = resource_manager_.get_image_view( depth_usage ),
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .loadOp = info.depth_stencil->load_op,
            .storeOp = info.depth_stencil->store_op,
            .clearValue = info.depth_stencil->clear_value,
        };
    }

    VkRenderingInfoKHR rendering_info{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
        .renderArea = info.render_area,
        .layerCount = 1,
        .colorAttachmentCount = static_cast<uint32_t>( color_attachments.size() ),
        .pColorAttachments = color_attachments.data(),
        .pDepthAttachment = info.depth_stencil ? &depth_attachment : nullptr,
        .pStencilAttachment = nullptr,// Not supporting stencil for now
    };

    vkCmdBeginRenderingKHR( command_buffer_, &rendering_info );
    in_renderpass_ = true;
    return std::nullopt;
}

std::optional<std::string> CommandList::end_renderpass() {
    if ( !in_renderpass_ ) { return "Not in render pass"; }
    vkCmdEndRenderingKHR( command_buffer_ );
    in_renderpass_ = false;
    return std::nullopt;
}

void CommandList::pipeline_barrier( const VkDependencyInfo& dependency_info ) const {
    vkCmdPipelineBarrier2KHR( command_buffer_, &dependency_info );
}

}// namespace aloe