#include <aloe/core/Device.h>
#include <aloe/core/Swapchain.h>
#include <aloe/util/log.h>

#include <GLFW/glfw3.h>
#include <gtest/gtest.h>

class SwapchainTestFixture : public ::testing::Test {
protected:
    std::shared_ptr<aloe::MockLogger> mock_logger_;
    std::shared_ptr<aloe::Device> device_;

    void SetUp() override {
        mock_logger_ = std::make_shared<aloe::MockLogger>();
        aloe::set_logger( mock_logger_ );
        aloe::set_logger_level( aloe::LogLevel::Trace );

        device_ = aloe::Device::create_device( {} ).value();
    }
};

TEST_F( SwapchainTestFixture, SwapchainInitialization ) {
    {
        auto swapchain = aloe::Swapchain::create_swapchain( *device_, {} );
        EXPECT_TRUE( swapchain.has_value() );
    }

    EXPECT_EQ( device_->debug_info().num_warning_, 0 );
    EXPECT_EQ( device_->debug_info().num_error_, 0 );
}

TEST_F( SwapchainTestFixture, WindowResizes ) {
    {
        auto swapchain = *aloe::Swapchain::create_swapchain( *device_, {} );
        auto* window = swapchain->window();

        int width, height;
        glfwGetWindowSize( window, &width, &height );

        // Resize the window, from the basic width, down to zero on one axis.
        int i = 1;
        while ( !swapchain->poll_events() ) {
            const auto next_width = std::max( 0, width - ( 10 * i++ ) );
            glfwSetWindowSize( window, next_width, height );

            if ( next_width == 0 ) { glfwSetWindowShouldClose( window, GLFW_TRUE ); }
        }
    }

    EXPECT_EQ( device_->debug_info().num_warning_, 0 );
    EXPECT_EQ( device_->debug_info().num_error_, 0 );
}
