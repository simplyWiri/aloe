#include <aloe/core/Device.h>
#include <aloe/util/log.h>

#include <gtest/gtest.h>

class DeviceTestsFixture : public ::testing::Test {
protected:
    std::shared_ptr<aloe::MockLogger> mock_logger_;

    void SetUp() override {
        mock_logger_ = std::make_shared<aloe::MockLogger>();
        aloe::set_logger( mock_logger_ );
        aloe::set_logger_level( aloe::LogLevel::Warn );
    }

    void TearDown() override {
        auto& debug_info = aloe::Device::debug_info();
        EXPECT_EQ( debug_info.num_warning, 0 );
        EXPECT_EQ( debug_info.num_error, 0 );

        if ( debug_info.num_warning > 0 || debug_info.num_error > 0 ) {
            for ( const auto& [level, message] : mock_logger_->get_entries() ) { std::cerr << message << std::endl; }
        }
    }
};

TEST_F( DeviceTestsFixture, RequiredDebugExtensionsAndLayersPresent ) {
    EXPECT_NO_THROW( aloe::Device( {} ) );
}

TEST_F( DeviceTestsFixture, RequiredExtensionsAndLayersPresent ) {
    EXPECT_NO_THROW( aloe::Device( { .enable_validation = false } ) );
}

TEST_F( DeviceTestsFixture, RequiredDebugExtensionsAndLayersPresentHeadless ) {
    EXPECT_NO_THROW( aloe::Device( { .headless = false } ) );
}
