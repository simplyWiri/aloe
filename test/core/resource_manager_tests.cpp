#include <aloe/core/Device.h>
#include <aloe/core/ResourceManager.h>
#include <aloe/util/log.h>

#include <gtest/gtest.h>

#include <numeric>

class ResourceManagerTestFixture : public ::testing::Test {
protected:
    std::shared_ptr<aloe::MockLogger> mock_logger_;
    std::unique_ptr<aloe::Device> device_;
    std::unique_ptr<aloe::ResourceManager> resource_manager_;

    void SetUp() override {
        mock_logger_ = std::make_shared<aloe::MockLogger>();
        aloe::set_logger( mock_logger_ );
        aloe::set_logger_level( aloe::LogLevel::Warn );

        device_ = std::make_unique<aloe::Device>( aloe::DeviceSettings{ .enable_validation = true, .headless = true } );
        resource_manager_ = std::make_unique<aloe::ResourceManager>( *device_ );
    }

    void TearDown() override {
        resource_manager_.reset( nullptr );

        VmaTotalStatistics stats;
        vmaCalculateStatistics( device_->allocator(), &stats );

        // There should be zero dangling allocations
        EXPECT_EQ( stats.total.statistics.allocationCount, 0 );

        // No validation errors
        auto& debug_info = aloe::Device::debug_info();
        EXPECT_EQ( debug_info.num_warning, 0 );
        EXPECT_EQ( debug_info.num_error, 0 );

        if ( debug_info.num_warning > 0 || debug_info.num_error > 0 ) {
            for ( const auto& [level, message] : mock_logger_->get_entries() ) { std::cerr << message << std::endl; }
        }
    }
};

TEST_F( ResourceManagerTestFixture, CreateBuffer ) {
    const auto handle = resource_manager_->create_buffer( {
        .size = 1024,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .name = "TestBuffer",
    } );

    ASSERT_NE( handle.id, 0 );
    EXPECT_NE( resource_manager_->get_buffer( handle ), VK_NULL_HANDLE );
}

TEST_F( ResourceManagerTestFixture, BufferHandlesUnique ) {
    const auto first = resource_manager_->create_buffer( {
        .size = 1024,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .name = "TestBuffer",
    } );
    const auto second = resource_manager_->create_buffer( {
        .size = 1024,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .name = "TestBuffer2",
    } );

    EXPECT_NE( first, second );
    EXPECT_NE( resource_manager_->get_buffer( first ), VK_NULL_HANDLE );
    EXPECT_NE( resource_manager_->get_buffer( second ), VK_NULL_HANDLE );
}

TEST_F( ResourceManagerTestFixture, UploadsHostVisibleMemory ) {
    constexpr std::array data = { 1, 2, 3, 4, 5, 6, 7, 8 };
    std::array<int, 8> read_back_data;

    constexpr auto data_size = data.size() * sizeof( int );
    const auto buffer = resource_manager_->create_buffer( {
        .size = data_size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .memory_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .name = "TestBuffer",
    } );

    EXPECT_EQ( resource_manager_->upload_to_buffer( buffer, data.data(), data_size ), data_size );
    EXPECT_EQ( resource_manager_->read_from_buffer( buffer, read_back_data.data(), data_size ), data_size );
    EXPECT_EQ( data, read_back_data );
}

TEST_F( ResourceManagerTestFixture, UploadToFreedBuffer ) {
    constexpr std::array data = { 1, 2, 3, 4, 5, 6, 7, 8 };
    const auto buffer = resource_manager_->create_buffer( {
        .size = 1234,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .memory_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .name = "TestBuffer",
    } );

    resource_manager_->free_buffer( buffer );
    EXPECT_EQ( resource_manager_->upload_to_buffer( buffer, data.data(), data.size() ), 0 );
}

TEST_F( ResourceManagerTestFixture, UploadToHostOnlyBuffer ) {
    constexpr std::array data = { 1, 2, 3, 4, 5, 6, 7, 8 };
    const auto buffer = resource_manager_->create_buffer( {
        .size = 1234,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .memory_usage = VMA_MEMORY_USAGE_GPU_ONLY,
        .name = "TestBuffer",
    } );

    resource_manager_->free_buffer( buffer );
    EXPECT_EQ( resource_manager_->upload_to_buffer( buffer, data.data(), data.size() ), 0 );
}

TEST_F( ResourceManagerTestFixture, FreeBuffer ) {
    const auto handle = resource_manager_->create_buffer( {
        .size = 1024,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .name = "TestBuffer",
    } );

    EXPECT_NE( handle.id, 0 );
    EXPECT_NE( resource_manager_->get_buffer( handle ), VK_NULL_HANDLE );

    resource_manager_->free_buffer( handle );
    EXPECT_EQ( resource_manager_->get_buffer( handle ), VK_NULL_HANDLE );
}

TEST_F( ResourceManagerTestFixture, CreateImage ) {
    const auto handle = resource_manager_->create_image( {
        .extent = { 1024, 1024, 1 },
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
        .name = "TestImage",
    } );

    EXPECT_NE( handle.id, 0 );
    EXPECT_NE( resource_manager_->get_image( handle ), VK_NULL_HANDLE );
}

TEST_F( ResourceManagerTestFixture, ImageHandlesUnique ) {
    const auto first = resource_manager_->create_image( {
        .extent = { 128, 128, 1 },
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
        .name = "TestImage",
    } );
    const auto second = resource_manager_->create_image( {
        .extent = { 128, 128, 1 },
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
        .name = "TestImage",
    } );

    EXPECT_NE( first, second );
    EXPECT_NE( resource_manager_->get_image( first ), VK_NULL_HANDLE );
    EXPECT_NE( resource_manager_->get_image( second ), VK_NULL_HANDLE );
}

TEST_F( ResourceManagerTestFixture, FreeImage ) {
    const auto handle = resource_manager_->create_image( {
        .extent = { 1024, 1024, 1 },
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
        .name = "TestImage",
    } );

    EXPECT_NE( handle.id, 0 );
    EXPECT_NE( resource_manager_->get_image( handle ), VK_NULL_HANDLE );

    resource_manager_->free_image( handle );
    EXPECT_EQ( resource_manager_->get_image( handle ), VK_NULL_HANDLE );
}

TEST_F( ResourceManagerTestFixture, StressTest ) {
    constexpr size_t num_buffers = 100;
    constexpr size_t num_images = 100;
    constexpr VkDeviceSize buffer_size = 256;

    std::vector<aloe::BufferHandle> buffer_handles;
    std::vector<aloe::ImageHandle> image_handles;

    // Test pattern data
    std::array<uint32_t, buffer_size / sizeof( uint32_t )> upload_data;
    std::iota( upload_data.begin(), upload_data.end(), 0 );

    // === Allocate buffers and upload/read data ===
    for ( size_t i = 0; i < num_buffers; ++i ) {
        const auto handle = resource_manager_->create_buffer( {
            .size = buffer_size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .memory_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .name = "StressBuffer",
        } );

        EXPECT_NE( handle.id, 0 );
        buffer_handles.push_back( handle );

        // Upload and read-back to validate integrity
        EXPECT_EQ( resource_manager_->upload_to_buffer( handle, upload_data.data(), buffer_size ), buffer_size );

        std::array<uint32_t, upload_data.size()> read_back{};
        EXPECT_EQ( resource_manager_->read_from_buffer( handle, read_back.data(), buffer_size ), buffer_size );
        EXPECT_EQ( upload_data, read_back );

        const auto image_handle = resource_manager_->create_image( {
            .extent = { 16, 16, 1 },
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
            .name = "StressImage",
        } );

        EXPECT_NE( image_handle.id, 0 );
        image_handles.push_back( image_handle );
    }

    VmaTotalStatistics stats;
    vmaCalculateStatistics( device_->allocator(), &stats );
    EXPECT_EQ( stats.total.statistics.allocationCount, num_buffers + num_images );
    constexpr auto total_buffer_size_bytes = ( num_buffers * buffer_size ) / sizeof( uint8_t );
    constexpr auto total_image_size_bytes = ( num_images * 16 * 16 * 4 ) / sizeof( uint8_t );
    EXPECT_EQ( stats.total.statistics.allocationBytes, total_buffer_size_bytes + total_image_size_bytes );

    // Free every second buffer backwards
    for ( size_t i = 1; i < buffer_handles.size(); i += 2 ) {
        resource_manager_->free_buffer( buffer_handles[buffer_handles.size() - i] );
    }

    // Free 3/4 buffers walking forwards
    for ( size_t i = 0; i < image_handles.size(); ++i ) {
        if ( i % 4 == 0 ) { continue; };
        resource_manager_->free_image( image_handles[i] );
    }

    // The destructor should free the remaining images and buffers.
}
