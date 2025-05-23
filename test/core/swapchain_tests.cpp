#include <aloe/core/Device.h>
#include <aloe/core/Swapchain.h>
#include <aloe/util/log.h>

#include <GLFW/glfw3.h>
#include <gtest/gtest.h>

#include <thread>

class SwapchainTestFixture : public ::testing::Test {
protected:
    std::shared_ptr<aloe::MockLogger> mock_logger_;
    std::shared_ptr<aloe::Device> device_;

    VkCommandPool pool_;
    VkCommandBuffer cmd_;

    // Additional helper methods
    VkSemaphore create_semaphore() {
        VkSemaphore semaphore;
        VkSemaphoreCreateInfo semaphore_info{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VkResult result = vkCreateSemaphore( device_->device(), &semaphore_info, nullptr, &semaphore );
        EXPECT_EQ( result, VK_SUCCESS );
        return semaphore;
    }

    void destroy_semaphore( VkSemaphore semaphore ) { vkDestroySemaphore( device_->device(), semaphore, nullptr ); }

    void transition_to_presentable( VkImage image ) {
        VkCommandBufferBeginInfo begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer( cmd_, &begin_info );

        VkImageMemoryBarrier2KHR image_barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR,
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };

        VkDependencyInfoKHR dep_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &image_barrier,
        };

        vkCmdPipelineBarrier2KHR( cmd_, &dep_info );
        vkEndCommandBuffer( cmd_ );
    }

    void SetUp() override {
        mock_logger_ = std::make_shared<aloe::MockLogger>();
        aloe::set_logger( mock_logger_ );
        aloe::set_logger_level( aloe::LogLevel::Warn );

        device_ = std::make_shared<aloe::Device>( aloe::DeviceSettings{} );

        VkCommandPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = device_->find_queues( VK_QUEUE_GRAPHICS_BIT ).front().family_index,
        };

        vkCreateCommandPool( device_->device(), &pool_info, nullptr, &pool_ );

        VkCommandBufferAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool_,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        vkAllocateCommandBuffers( device_->device(), &alloc_info, &cmd_ );
    }

    void TearDown() override {
        vkDeviceWaitIdle( device_->device() );
        vkDestroyCommandPool( device_->device(), pool_, nullptr );

        auto& debug_info = aloe::Device::debug_info();
        EXPECT_EQ( debug_info.num_warning, 0 );
        EXPECT_EQ( debug_info.num_error, 0 );

        if ( debug_info.num_warning > 0 || debug_info.num_error > 0 ) {
            for ( const auto& [level, message] : mock_logger_->get_entries() ) { std::cerr << message << std::endl; }
        }
    }
};

TEST_F( SwapchainTestFixture, SwapchainInitialization ) {
    EXPECT_NO_THROW( device_->make_swapchain( {} ) );
}

TEST_F( SwapchainTestFixture, WindowResizes ) {
    auto swapchain = device_->make_swapchain( {} );
    auto* window = swapchain->window();

    int width, height;
    glfwGetWindowSize( window, &width, &height );

    int fb_width, fb_height;
    glfwGetFramebufferSize( window, &fb_width, &fb_height );

    VkExtent2D initial_extent = swapchain->get_extent();
    EXPECT_EQ( initial_extent.width, static_cast<uint32_t>( fb_width ) );
    EXPECT_EQ( initial_extent.height, static_cast<uint32_t>( fb_height ) );

    int i = 1;
    while ( !swapchain->poll_events() ) {
        const auto next_width = std::max( 0, width - ( 10 * i++ ) );
        glfwSetWindowSize( window, next_width, height );

        // After resize, extent should match framebuffer size (unless minimized)
        if ( next_width > 0 ) {
            glfwGetFramebufferSize( window, &fb_width, &fb_height );

            const VkExtent2D extent = swapchain->get_extent();
            EXPECT_EQ( extent.width, static_cast<uint32_t>( fb_width ) );
            EXPECT_EQ( extent.height, static_cast<uint32_t>( fb_height ) );
        } else if ( next_width == 0 ) {
            glfwSetWindowShouldClose( window, GLFW_TRUE );
        }
    }
}

TEST_F( SwapchainTestFixture, AcquireNextImageSucceeds ) {
    auto swapchain = device_->make_swapchain( {} );
    VkSemaphore image_avail_semaphore = create_semaphore();

    // Call acquire_next_image
    const auto render_target = swapchain->acquire_next_image( image_avail_semaphore );
    ASSERT_TRUE( render_target.has_value() ) << "Swapchain::acquire_next_image failed";
    EXPECT_NE( render_target->image, VK_NULL_HANDLE );
    EXPECT_NE( render_target->view, VK_NULL_HANDLE );

    destroy_semaphore( image_avail_semaphore );
}

TEST_F( SwapchainTestFixture, AcquireNextImageFailsDueToResizeSucceeds ) {
    auto swapchain = device_->make_swapchain( {} );

    int width, height;
    glfwGetWindowSize( swapchain->window(), &width, &height );
    glfwSetWindowSize( swapchain->window(), 0, 0 );

    {
        const auto image_avail_semaphore = create_semaphore();
        if ( swapchain->acquire_next_image( image_avail_semaphore ) == std::nullopt ) {
            // If we fail to acquire the render target, we need to ensure that after it has been fixed, we can re-acquire with
            // the same Semaphore.

            glfwSetWindowSize( swapchain->window(), width, height );
            swapchain->poll_events();

            const auto render_target = swapchain->acquire_next_image( image_avail_semaphore );
            ASSERT_TRUE( render_target.has_value() ) << "Failed to acquire the image";

            // Extent should match restored framebuffer size
            {
                int fb_width, fb_height;
                glfwGetFramebufferSize( swapchain->window(), &fb_width, &fb_height );

                const VkExtent2D extent = swapchain->get_extent();
                EXPECT_EQ( extent.width, static_cast<uint32_t>( fb_width ) );
                EXPECT_EQ( extent.height, static_cast<uint32_t>( fb_height ) );
            }
        }

        vkDeviceWaitIdle( device_->device() );
        destroy_semaphore( image_avail_semaphore );
    }
}

TEST_F( SwapchainTestFixture, PresentSucceeds ) {
    auto swapchain = device_->make_swapchain( {} );
    VkSemaphore image_avail_semaphore = create_semaphore();
    VkSemaphore render_finished_semaphore = create_semaphore();

    // Acquire an image first
    const auto render_target = swapchain->acquire_next_image( image_avail_semaphore );
    ASSERT_TRUE( render_target.has_value() ) << "Failed to acquire the image";

    // Retrieve graphics queue for presentation
    const auto queue = device_->find_queues( VK_QUEUE_GRAPHICS_BIT ).front();

    transition_to_presentable( render_target->image );

    // You would normally perform rendering operations here and signal render_finished_semaphore
    // For testing purposes, we'll submit an empty batch just to signal the semaphore immediately:
    VkPipelineStageFlags wait_stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &image_avail_semaphore,
        .pWaitDstStageMask = &wait_stages,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd_,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &render_finished_semaphore,
    };

    VkResult submit_result = vkQueueSubmit( queue.queue, 1, &submit_info, VK_NULL_HANDLE );
    EXPECT_EQ( submit_result, VK_SUCCESS ) << "Failed to submit empty queue";

    vkQueueWaitIdle( queue.queue );

    // Then call present using the semaphore
    VkResult present_result = swapchain->present( queue.queue, render_finished_semaphore );
    EXPECT_TRUE( present_result == VK_SUCCESS ) << "Presentation failed unexpectedly";

    vkDeviceWaitIdle( device_->device() );

    destroy_semaphore( image_avail_semaphore );
    destroy_semaphore( render_finished_semaphore );
}

TEST_F( SwapchainTestFixture, PresentFailsDueToResize ) {
    auto swapchain = device_->make_swapchain( {} );
    auto* window = swapchain->window();

    VkSemaphore image_avail_semaphore = create_semaphore();
    VkSemaphore render_finished_semaphore = create_semaphore();

    // Acquire an image first
    const auto render_target = swapchain->acquire_next_image( image_avail_semaphore );
    ASSERT_TRUE( render_target.has_value() ) << "Failed to acquire the image";

    // Retrieve graphics queue for presentation
    const auto queue = device_->find_queues( VK_QUEUE_GRAPHICS_BIT ).front();

    transition_to_presentable( render_target->image );

    // You would normally perform rendering operations here and signal render_finished_semaphore
    // For testing purposes, we'll submit an empty batch just to signal the semaphore immediately:
    VkPipelineStageFlags wait_stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &image_avail_semaphore,
        .pWaitDstStageMask = &wait_stages,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd_,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &render_finished_semaphore,
    };

    VkResult submit_result = vkQueueSubmit( queue.queue, 1, &submit_info, VK_NULL_HANDLE );
    EXPECT_EQ( submit_result, VK_SUCCESS ) << "Failed to submit empty queue";

    vkQueueWaitIdle( queue.queue );

    int width, height;
    glfwGetWindowSize( window, &width, &height );
    glfwSetWindowSize( window, 0, 0 );

    // Then call present using the semaphore
    VkResult present_result = swapchain->present( queue.queue, render_finished_semaphore );
    EXPECT_TRUE( present_result != VK_SUCCESS ) << "Presentation succeeded unexpectedly";

    vkDeviceWaitIdle( device_->device() );

    destroy_semaphore( image_avail_semaphore );
    destroy_semaphore( render_finished_semaphore );
}
