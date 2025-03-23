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
    auto swapchain = aloe::Swapchain::create_swapchain( *device_, {} );
    EXPECT_TRUE( swapchain.has_value() );

    EXPECT_EQ( device_->debug_info().num_warning_, 0 );
    EXPECT_EQ( device_->debug_info().num_error_, 0 );
}
