#include <aloe/core/Device.h>
#include <aloe/core/ResourceManager.h>
#include <aloe/util/log.h>

#include <gtest/gtest.h>

class ResourceManagerTestFixture : public ::testing::Test {
protected:
    std::shared_ptr<aloe::MockLogger> mock_logger_;
    std::unique_ptr<aloe::Device> device_;
    std::unique_ptr<aloe::ResourceManager> resource_manager_;

    void SetUp() override {
        mock_logger_ = std::make_shared<aloe::MockLogger>();
        aloe::set_logger( mock_logger_ );
        aloe::set_logger_level( aloe::LogLevel::Trace );

        device_ = aloe::Device::create_device( { .enable_validation = true, .headless = true } ).value();
        resource_manager_ = std::make_unique<aloe::ResourceManager>( device_->allocator() );
    }

    void TearDown() override {
        resource_manager_.reset(nullptr);

        VmaTotalStatistics stats;
        vmaCalculateStatistics( device_->allocator(), &stats );

        // There should be zero dangling allocations
        EXPECT_EQ( stats.total.statistics.allocationCount, 0 );

        // No validation errors
        const auto debug_info = aloe::Device::debug_info();
        EXPECT_EQ( debug_info.num_warning, 0 );
        EXPECT_EQ( debug_info.num_error, 0 );
    }
};

TEST_F( ResourceManagerTestFixture, CreateBuffer ) {
    const auto handle = resource_manager_->create_buffer( { .size = 1024, .debug_name = "TestBuffer" } );

    ASSERT_NE( handle.id, 0 );
    EXPECT_NE(resource_manager_->get_buffer( handle ), VK_NULL_HANDLE);
}

TEST_F( ResourceManagerTestFixture, BufferHandlesUnique ) {
    const auto first = resource_manager_->create_buffer( { .size = 1024, .debug_name = "TestBuffer" } );
    const auto second = resource_manager_->create_buffer( { .size = 1024, .debug_name = "TestBuffer" } );

    EXPECT_NE( first, second );
    EXPECT_NE(resource_manager_->get_buffer( first ), VK_NULL_HANDLE);
    EXPECT_NE(resource_manager_->get_buffer( second ), VK_NULL_HANDLE);
}

TEST_F( ResourceManagerTestFixture, CreateImage ) {
    const auto handle = resource_manager_->create_image( {
        .extent = { 1024, 1024, 1 },
        .format = VK_FORMAT_R8G8B8A8_UNORM,
    } );

    EXPECT_NE( handle.id, 0 );
    EXPECT_NE(resource_manager_->get_image( handle ), VK_NULL_HANDLE);
}

TEST_F( ResourceManagerTestFixture, ImageHandlesUnique ) {
    const auto first = resource_manager_->create_image( {
        .extent = { 128, 128, 1 },
        .format = VK_FORMAT_R8G8B8A8_UNORM,
    } );
    const auto second = resource_manager_->create_image( {
        .extent = { 128, 128, 1 },
        .format = VK_FORMAT_R8G8B8A8_UNORM,
    } );

    EXPECT_NE( first, second );
    EXPECT_NE(resource_manager_->get_image( first ), VK_NULL_HANDLE);
    EXPECT_NE(resource_manager_->get_image( second ), VK_NULL_HANDLE);
}

