#include <aloe/core/Device.h>
#include <aloe/util/log.h>

#include <gtest/gtest.h>

class DeviceTestsFixture : public ::testing::Test {
protected:
    std::shared_ptr<aloe::MockLogger> mock_logger_;

    void SetUp() override {
        mock_logger_ = std::make_shared<aloe::MockLogger>();
        aloe::set_logger( mock_logger_ );
        aloe::set_logger_level( aloe::LogLevel::Trace );
    }
};

TEST_F( DeviceTestsFixture, RequiredDebugExtensionsAndLayersPresent ) {
    auto device = aloe::Device::create_device( {} );
    EXPECT_TRUE( device.has_value() );

    const auto debug_info = device.value()->debug_info();
    EXPECT_EQ( debug_info.num_warning_, 0 );
    EXPECT_EQ( debug_info.num_error_, 0 );
}

TEST_F( DeviceTestsFixture, RequiredExtensionsAndLayersPresent ) {
    auto device = aloe::Device::create_device( { .enable_validation = false } );
    EXPECT_TRUE( device.has_value() );

    const auto debug_info = device.value()->debug_info();
    EXPECT_EQ( debug_info.num_warning_, 0 );
    EXPECT_EQ( debug_info.num_error_, 0 );
}

TEST_F( DeviceTestsFixture, RequiredDebugExtensionsAndLayersPresentHeadless ) {
    auto device = aloe::Device::create_device( { .headless = true } );
    EXPECT_TRUE( device.has_value() );

    const auto debug_info = device.value()->debug_info();
    EXPECT_EQ( debug_info.num_warning_, 0 );
    EXPECT_EQ( debug_info.num_error_, 0 );
}
