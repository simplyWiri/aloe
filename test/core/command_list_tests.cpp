#include <gtest/gtest.h>
#include <aloe/core/CommandList.h>
#include <aloe/core/Device.h>
#include <aloe/core/PipelineManager.h>
#include <aloe/core/ResourceManager.h>

namespace {

class CommandListTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = std::make_unique<aloe::Device>(aloe::DeviceSettings{
            .name = "CommandListTest",
            .enable_validation = true
        });
        pipeline_mgr_ = device_->make_pipeline_manager({"shaders"});
        resource_mgr_ = device_->make_resource_manager();
        
        // Create a command buffer for testing
        VkCommandPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = device_->queues_by_capability(VK_QUEUE_GRAPHICS_BIT).front().family_index
        };
        VK_CHECK(vkCreateCommandPool(device_->handle(), &pool_info, nullptr, &cmd_pool_));
        
        VkCommandBufferAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = cmd_pool_,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };
        VK_CHECK(vkAllocateCommandBuffers(device_->handle(), &alloc_info, &cmd_));
        
        VkCommandBufferBeginInfo begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        };
        VK_CHECK(vkBeginCommandBuffer(cmd_, &begin_info));
        
        // Create test resources
        render_target_ = resource_mgr_->create_image({
            .extent = {64, 64, 1},
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .name = "test_render_target"
        });
        
        // Create test pipelines
        compute_pipeline_ = pipeline_mgr_->compile_pipeline(aloe::ComputePipelineInfo{
            .compute_shader = {"basic_compute.comp"}
        }).value();

        // Create command list for testing
        cmd_list_ = std::make_unique<aloe::CommandList>(cmd_, *pipeline_mgr_, *resource_mgr_);
    }
    
    void TearDown() override {
        vkEndCommandBuffer(cmd_);
        vkDestroyCommandPool(device_->handle(), cmd_pool_, nullptr);
        cmd_list_.reset();
        pipeline_mgr_.reset();
        resource_mgr_.reset();
        device_.reset();
    }
    
    std::unique_ptr<aloe::Device> device_;
    std::shared_ptr<aloe::PipelineManager> pipeline_mgr_;
    std::shared_ptr<aloe::ResourceManager> resource_mgr_;
    VkCommandPool cmd_pool_{VK_NULL_HANDLE};
    VkCommandBuffer cmd_{VK_NULL_HANDLE};
    std::unique_ptr<aloe::CommandList> cmd_list_;
    
    aloe::ImageHandle render_target_;
    aloe::PipelineHandle compute_pipeline_;
    aloe::PipelineHandle graphics_pipeline_;
};

TEST_F(CommandListTest, PipelineBindingScope) {
    // Binding compute pipeline outside renderpass should work
    {
        auto scope = cmd_list_->bind_pipeline(compute_pipeline_);
        scope.dispatch(1, 1, 1);
    }
    
    // Binding graphics pipeline inside renderpass should work
    aloe::RenderingInfo render_info{
        .colors = {{
            .image = render_target_,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .store_op = VK_ATTACHMENT_STORE_OP_STORE,
            .clear_value = {0.0f, 0.0f, 0.0f, 1.0f}
        }},
        .render_area = {{0, 0}, {64, 64}}
    };
    
    cmd_list_->begin_renderpass(render_info);
    {
        auto scope = cmd_list_->bind_pipeline(graphics_pipeline_);
        scope.draw(3, 1, 0, 0);
    }
    cmd_list_->end_renderpass();
}

TEST_F(CommandListTest, ValidateRenderpassState) {
    // Cannot bind compute pipeline inside renderpass
    aloe::RenderingInfo render_info{
        .colors = {{
            .image = render_target_,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .store_op = VK_ATTACHMENT_STORE_OP_STORE,
            .clear_value = {0.0f, 0.0f, 0.0f, 1.0f}
        }},
        .render_area = {{0, 0}, {64, 64}}
    };
    
    cmd_list_->begin_renderpass(render_info);
    EXPECT_THROW(cmd_list_->bind_pipeline(compute_pipeline_), std::runtime_error);
    cmd_list_->end_renderpass();
    
    // Cannot draw outside renderpass
    {
        auto scope = cmd_list_->bind_pipeline(graphics_pipeline_);
        EXPECT_THROW(scope.draw(3, 1, 0, 0), std::runtime_error);
    }
}

TEST_F(CommandListTest, ValidatePipelineTypeCommands) {
    // Cannot dispatch with graphics pipeline
    {
        auto scope = cmd_list_->bind_pipeline(graphics_pipeline_);
        EXPECT_THROW(scope.dispatch(1, 1, 1), std::runtime_error);
    }
    
    // Cannot draw with compute pipeline (outside renderpass)
    {
        auto scope = cmd_list_->bind_pipeline(compute_pipeline_);
        EXPECT_THROW(scope.draw(3, 1, 0, 0), std::runtime_error);
    }
}

TEST_F(CommandListTest, DebugMarkers) {
    cmd_list_->begin_debug_marker("Test Marker 1");
    cmd_list_->begin_debug_marker("Test Marker 2");
    cmd_list_->end_debug_marker();
    cmd_list_->end_debug_marker();
    
    // Cannot end more markers than began
    EXPECT_THROW(cmd_list_->end_debug_marker(), std::runtime_error);
}

} // namespace