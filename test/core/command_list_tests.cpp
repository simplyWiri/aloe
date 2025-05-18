#include <aloe/core/CommandList.h>
#include <aloe/core/Device.h>
#include <aloe/core/PipelineManager.h>
#include <aloe/core/ResourceManager.h>
#include <aloe/util/log.h>

#include <gtest/gtest.h>

using namespace std::chrono_literals;

class CommandListTestFixture : public ::testing::Test {
protected:
    std::shared_ptr<aloe::MockLogger> mock_logger_;
    std::unique_ptr<aloe::Device> device_;
    std::shared_ptr<aloe::PipelineManager> pipeline_manager_;
    std::shared_ptr<aloe::ResourceManager> resource_manager_;
    VkCommandBuffer command_buffer_;
    VkCommandPool command_pool_;

    aloe::SimulationState sim_state_{ .sim_index = 0, .time_since_epoch = 0us, .delta_time = 0us };

    void SetUp() override {
        mock_logger_ = std::make_shared<aloe::MockLogger>();
        aloe::set_logger( mock_logger_ );
        aloe::set_logger_level( aloe::LogLevel::Warn );

        device_ = std::make_unique<aloe::Device>( aloe::DeviceSettings{ .enable_validation = true, .headless = true } );
        resource_manager_ = device_->make_resource_manager();
        pipeline_manager_ = device_->make_pipeline_manager( {} );

        // Create a command pool and allocate a command buffer
        VkCommandPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = device_->find_queues( VK_QUEUE_GRAPHICS_BIT )[0].family_index,
        };

        vkCreateCommandPool( device_->device(), &pool_info, nullptr, &command_pool_ );

        VkCommandBufferAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = command_pool_,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        vkAllocateCommandBuffers( device_->device(), &alloc_info, &command_buffer_ );
        VkCommandBufferBeginInfo begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        };
        vkBeginCommandBuffer( command_buffer_, &begin_info );
    }

    void TearDown() override {
        vkEndCommandBuffer( command_buffer_ );

        if ( command_pool_ != VK_NULL_HANDLE ) { vkDestroyCommandPool( device_->device(), command_pool_, nullptr ); }

        resource_manager_.reset();
        pipeline_manager_.reset();
        device_.reset();

        auto& debug_info = aloe::Device::debug_info();
        EXPECT_EQ( debug_info.num_warning, 0 );
        EXPECT_EQ( debug_info.num_error, 0 );

        if ( debug_info.num_warning > 0 || debug_info.num_error > 0 ) {
            for ( const auto& [level, message] : mock_logger_->get_entries() ) { std::cerr << message << std::endl; }
        }
    }

    aloe::PipelineHandle create_compute_pipeline() {
        const auto shader_str = R"(
            [shader("compute")]
            void compute_main() { }
        )";

        pipeline_manager_->set_virtual_file( "compute.slang", shader_str );

        const auto result = pipeline_manager_->compile_pipeline( { .compute_shader = {
                                                                       .name = "compute.slang",
                                                                       .entry_point = "compute_main",
                                                                   } } );
        EXPECT_TRUE( result.has_value() );
        return result.value();
    }

    aloe::PipelineHandle create_graphics_pipeline() {
        const auto shader_str = R"(
            [shader("vertex")]
            void vertex_main() { }
            [shader("fragment")]
            void fragment_main() { }
        )";

        pipeline_manager_->set_virtual_file( "graphics.slang", shader_str );

        const auto result = pipeline_manager_->compile_pipeline( {
            .vertex_shader = { .name = "graphics.slang", .entry_point = "vertex_main" },
            .fragment_shader = { .name = "graphics.slang", .entry_point = "fragment_main" },
        } );
        EXPECT_TRUE( result.has_value() );
        return result.value();
    }
};

//------------------------------------------------------------------------------
// Construction & State Tests
//------------------------------------------------------------------------------

TEST_F( CommandListTestFixture, ConstructCommandList_Succeeds ) {
    ASSERT_NO_THROW( {
        aloe::CommandList cmd_list( *pipeline_manager_,
                                    *resource_manager_,
                                    "Test Section",
                                    command_buffer_,
                                    sim_state_ );
    } );
}

TEST_F( CommandListTestFixture, SimulationStateAccess_MatchesInput ) {
    aloe::CommandList cmd_list( *pipeline_manager_, *resource_manager_, "Test Section", command_buffer_, sim_state_ );
    const auto& state = cmd_list.state();
    EXPECT_EQ( state.sim_index, sim_state_.sim_index );
    EXPECT_EQ( state.time_since_epoch, sim_state_.time_since_epoch );
    EXPECT_EQ( state.delta_time, sim_state_.delta_time );
}

//------------------------------------------------------------------------------
// Render Pass Usage Tests
//------------------------------------------------------------------------------

TEST_F( CommandListTestFixture, BeginRenderPass_NestedBeginFails ) {
    aloe::CommandList cmd_list( *pipeline_manager_, *resource_manager_, "Test Section", command_buffer_, sim_state_ );
    auto test_image = resource_manager_->create_image( {
        .extent = { 64, 64, 1 },
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .name = "test_image",
    } );
    resource_manager_->bind_resource( aloe::usage( test_image, aloe::ColorAttachmentWrite ) );

    aloe::RenderingInfo render_info{
        .colors = { {
            .image = test_image,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .store_op = VK_ATTACHMENT_STORE_OP_STORE,
            .clear_value = {},
        } },
        .render_area = { { 0, 0 }, { 64, 64 } },
    };
    ASSERT_FALSE( cmd_list.begin_renderpass( render_info ).has_value() );
    auto result = cmd_list.begin_renderpass( render_info );
    EXPECT_TRUE( result.has_value() );
    EXPECT_EQ( result.value(), "Already in render pass" );
    ASSERT_FALSE( cmd_list.end_renderpass().has_value() );
}

TEST_F( CommandListTestFixture, EndRenderPass_WithoutBeginFails ) {
    aloe::CommandList cmd_list( *pipeline_manager_, *resource_manager_, "Test Section", command_buffer_, sim_state_ );
    auto result = cmd_list.end_renderpass();
    EXPECT_TRUE( result.has_value() );
    EXPECT_EQ( result.value(), "Not in render pass" );
}

TEST_F( CommandListTestFixture, EndRenderPass_DoubleEndFails ) {
    aloe::CommandList cmd_list( *pipeline_manager_, *resource_manager_, "Test Section", command_buffer_, sim_state_ );
    auto test_image = resource_manager_->create_image( {
        .extent = { 64, 64, 1 },
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .name = "test_image",
    } );
    resource_manager_->bind_resource( aloe::usage( test_image, aloe::ColorAttachmentWrite ) );
    aloe::RenderingInfo render_info{
        .colors = { {
            .image = test_image,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .store_op = VK_ATTACHMENT_STORE_OP_STORE,
            .clear_value = {},
        } },
        .render_area = { { 0, 0 }, { 64, 64 } },
    };
    ASSERT_FALSE( cmd_list.begin_renderpass( render_info ).has_value() );
    ASSERT_FALSE( cmd_list.end_renderpass().has_value() );
    auto result = cmd_list.end_renderpass();
    EXPECT_TRUE( result.has_value() );
    EXPECT_EQ( result.value(), "Not in render pass" );
}

TEST_F( CommandListTestFixture, UnendedRenderpassAsserts ) {
    auto mock_logger = std::make_shared<aloe::MockLogger>();
    aloe::set_logger( mock_logger );

    {
        aloe::CommandList cmd_list( *pipeline_manager_,
                                    *resource_manager_,
                                    "Test Section",
                                    command_buffer_,
                                    sim_state_ );

        auto test_image = resource_manager_->create_image( {
            .extent = { 64, 64, 1 },
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .name = "test_image",
        } );

        resource_manager_->bind_resource( aloe::usage( test_image, aloe::ColorAttachmentWrite ) );

        aloe::RenderingInfo render_info{
            .colors = { {
                .image = test_image,
                .format = VK_FORMAT_R8G8B8A8_UNORM,
                .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .store_op = VK_ATTACHMENT_STORE_OP_STORE,
                .clear_value = {},
            } },
            .render_area = { { 0, 0 }, { 64, 64 } },
        };

        EXPECT_FALSE( cmd_list.begin_renderpass( render_info ).has_value() );
        EXPECT_TRUE( cmd_list.in_renderpass() );
        // Let cmd_list be destroyed while in renderpass
    }

    // Verify that an error was logged
    const auto& entries = mock_logger->get_entries();
    EXPECT_EQ( entries.size(), 1 );
    EXPECT_EQ( entries[0].level, aloe::LogLevel::Error );
    EXPECT_EQ( entries[0].message, "Renderpass was not ended before CommandList destruction" );
}

TEST_F( CommandListTestFixture, InvalidPipelineBindAsserts ) {
    aloe::CommandList cmd_list( *pipeline_manager_, *resource_manager_, "Test Section", command_buffer_, sim_state_ );

    aloe::PipelineHandle invalid_handle{ 543 };
    auto result = cmd_list.bind_pipeline( invalid_handle ).draw( 3, 1, 0, 0 );
    EXPECT_TRUE( result.has_value() ) << "Expected invalid pipeline bind to fail";
}

TEST_F( CommandListTestFixture, DrawWithComputePipelineFails ) {
    aloe::CommandList cmd_list( *pipeline_manager_, *resource_manager_, "Test Section", command_buffer_, sim_state_ );

    auto compute_pipeline = create_compute_pipeline();
    auto result = cmd_list.bind_pipeline( compute_pipeline ).draw( 3, 1, 0, 0 );
    EXPECT_TRUE( result.has_value() ) << "Expected draw with compute pipeline to fail";
    EXPECT_EQ( result.value(), "Cannot draw with a compute pipeline" );
}

TEST_F( CommandListTestFixture, DispatchWithGraphicsPipelineFails ) {
    aloe::CommandList cmd_list( *pipeline_manager_, *resource_manager_, "Test Section", command_buffer_, sim_state_ );

    auto graphics_pipeline = create_graphics_pipeline();
    auto result = cmd_list.bind_pipeline( graphics_pipeline ).dispatch( 8, 8, 1 );
    EXPECT_TRUE( result.has_value() ) << "Expected dispatch with graphics pipeline to fail";
    EXPECT_EQ( result.value(), "Cannot dispatch with a graphics pipeline" );
}

// todo: uncomment when graphics pipelines are supported.
// TEST_F( CommandListTestFixture, ValidRenderPassUsage ) {
//     aloe::CommandList cmd_list( *pipeline_manager_, *resource_manager_, "Test Section", command_buffer_, sim_state_ );
//
//     auto test_image = resource_manager_->create_image( { .extent = { 64, 64, 1 },
//                                                          .format = VK_FORMAT_R8G8B8A8_UNORM,
//                                                          .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
//                                                          .name = "test_image" } );
//     resource_manager_->bind_resource( aloe::usage( test_image, aloe::ColorAttachmentWrite ) );
//
//     aloe::RenderingInfo render_info{ .colors = { { .image = test_image,
//                                                    .format = VK_FORMAT_R8G8B8A8_UNORM,
//                                                    .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
//                                                    .store_op = VK_ATTACHMENT_STORE_OP_STORE,
//                                                    .clear_value = {} } },
//                                      .render_area = { { 0, 0 }, { 64, 64 } } };
//
//     auto result = cmd_list.begin_renderpass( render_info );
//     EXPECT_FALSE( result.has_value() ) << "Expected begin_renderpass to succeed";
//     EXPECT_TRUE( cmd_list.in_renderpass() );
//
//     auto graphics_pipeline = create_graphics_pipeline();
//     auto draw_result = cmd_list.bind_pipeline( graphics_pipeline ).draw( 3, 1, 0, 0 );
//     EXPECT_FALSE( draw_result.has_value() ) << "Expected draw to succeed";
//
//     result = cmd_list.end_renderpass();
//     EXPECT_FALSE( result.has_value() ) << "Expected end_renderpass to succeed";
//     EXPECT_FALSE( cmd_list.in_renderpass() );
// }

TEST_F( CommandListTestFixture, DrawOutsideRenderPassFails ) {
    aloe::CommandList cmd_list( *pipeline_manager_, *resource_manager_, "Test Section", command_buffer_, sim_state_ );

    auto graphics_pipeline = create_graphics_pipeline();
    auto result = cmd_list.bind_pipeline( graphics_pipeline ).draw( 3, 1, 0, 0 );
    EXPECT_TRUE( result.has_value() ) << "Expected draw outside renderpass to fail";
    EXPECT_EQ( result.value(), "Cannot draw outside of a render pass" );
}

TEST_F( CommandListTestFixture, DispatchInRenderPassFails ) {
    aloe::CommandList cmd_list( *pipeline_manager_, *resource_manager_, "Test Section", command_buffer_, sim_state_ );

    auto test_image = resource_manager_->create_image( {
        .extent = { 64, 64, 1 },
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .name = "test_image",
    } );
    resource_manager_->bind_resource( aloe::usage( test_image, aloe::ColorAttachmentWrite ) );

    aloe::RenderingInfo render_info{
        .colors = { {
            .image = test_image,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .store_op = VK_ATTACHMENT_STORE_OP_STORE,
            .clear_value = {},
        } },
        .render_area = { { 0, 0 }, { 64, 64 } },
    };

    auto compute_pipeline = create_compute_pipeline();
    EXPECT_FALSE( cmd_list.begin_renderpass( render_info ).has_value() );

    auto result = cmd_list.bind_pipeline( compute_pipeline ).dispatch( 8, 8, 1 );
    EXPECT_TRUE( result.has_value() ) << "Expected dispatch in renderpass to fail";
    EXPECT_EQ( result.value(), "Cannot dispatch inside a render pass" );

    EXPECT_FALSE( cmd_list.end_renderpass().has_value() );
}