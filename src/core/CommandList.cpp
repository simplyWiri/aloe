#include <aloe/core/CommandList.h>
#include <aloe/core/PipelineManager.h>
#include <aloe/core/ResourceManager.h>

namespace aloe {

BoundPipelineScope::BoundPipelineScope( CommandList& cmd, PipelineManager& pipeline_manager, PipelineHandle handle )
    : cmd_( cmd )
    , pipeline_manager_( pipeline_manager )
    , pipeline_( handle ) {
}

void BoundPipelineScope::dispatch( uint32_t x, uint32_t y, uint32_t z ) {
    pipeline_manager_.bind_pipeline( pipeline_, cmd_.cmd_ );
    vkCmdDispatch( cmd_.cmd_, x, y, z );
}

CommandList::CommandList( const char* task_name, VkCommandBuffer cmd, PipelineManager& pipeline_mgr, ResourceManager& resource_mgr )
    : cmd_( cmd )
    , pipeline_mgr_( pipeline_mgr )
    , resource_mgr_( resource_mgr ) {

    VkDebugUtilsLabelEXT label{ .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, .pLabelName = task_name };
    vkCmdBeginDebugUtilsLabelEXT( cmd_, &label );
}

CommandList::~CommandList() {
    vkCmdEndDebugUtilsLabelEXT( cmd_ );
}

BoundPipelineScope CommandList::bind_pipeline( PipelineHandle handle ) {
    return { *this, pipeline_mgr_, handle };
}


}// namespace aloe