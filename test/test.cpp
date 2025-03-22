#include <aloe/core/Device.h>
#include <gtest/gtest.h>

TEST(DeviceTests, RequiredDebugExtensionsAndLayersPresent) {
    auto device = aloe::Device::create_device( {} );
    EXPECT_TRUE(device.has_value());

    const auto debug_info = device.value()->debug_info();
    EXPECT_EQ(debug_info.num_warning_, 0);
    EXPECT_EQ(debug_info.num_error_, 0);
}

TEST(DeviceTests, RequiredExtensionsAndLayersPresent) {
    auto device = aloe::Device::create_device( { .enable_validation = false} );
    EXPECT_TRUE(device.has_value());

    const auto debug_info = device.value()->debug_info();
    EXPECT_EQ(debug_info.num_warning_, 0);
    EXPECT_EQ(debug_info.num_error_, 0);
}
