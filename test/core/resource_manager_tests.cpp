#include <aloe/core/Device.h>
#include <aloe/core/ResourceManager.h>
#include <aloe/util/log.h>

#include <gtest/gtest.h>

#include <numeric>

class ResourceManagerTestFixture : public ::testing::Test {
protected:
    std::shared_ptr<aloe::MockLogger> mock_logger_;
    std::unique_ptr<aloe::Device> device_;
    std::shared_ptr<aloe::ResourceManager> resource_manager_;

    void SetUp() override {
        mock_logger_ = std::make_shared<aloe::MockLogger>();
        aloe::set_logger( mock_logger_ );
        aloe::set_logger_level( aloe::LogLevel::Warn );

        device_ = std::make_unique<aloe::Device>( aloe::DeviceSettings{ .enable_validation = true, .headless = true } );
        resource_manager_ = device_->make_resource_manager();
    }

    void TearDown() override {
        resource_manager_.reset();
        device_.reset( nullptr );

        auto& debug_info = aloe::Device::debug_info();

        // No memory leaks
        EXPECT_EQ( debug_info.memory_stats_.total.statistics.allocationCount, 0 );

        // No validation errors
        EXPECT_EQ( debug_info.num_warning, 0 );
        EXPECT_EQ( debug_info.num_error, 0 );

        if ( debug_info.num_warning > 0 || debug_info.num_error > 0 ) {
            for ( const auto& [level, message] : mock_logger_->get_entries() ) { std::cerr << message << std::endl; }
        }
    }
};

//------------------------------------------------------------------------------
// Basic Creation/Destruction Tests
//------------------------------------------------------------------------------

TEST_F( ResourceManagerTestFixture, CreateBuffer_ReturnsValidHandle ) {
    const auto handle = resource_manager_->create_buffer( {
        .size = 1024,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .name = "TestBuffer",
    } );

    ASSERT_NE( handle.raw, 0 );
    EXPECT_NE( resource_manager_->get_buffer( handle ), VK_NULL_HANDLE );
}

TEST_F( ResourceManagerTestFixture, CreateBuffer_HandlesAreUnique ) {
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

TEST_F( ResourceManagerTestFixture, CreateImage_ReturnsValidHandle ) {
    const auto handle = resource_manager_->create_image( {
        .extent = { 1024, 1024, 1 },
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
        .name = "TestImage",
    } );

    ASSERT_NE( handle.raw, 0 );
    EXPECT_NE( resource_manager_->get_image( handle ), VK_NULL_HANDLE );
}

TEST_F( ResourceManagerTestFixture, CreateImage_HandlesAreUnique ) {
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

TEST_F( ResourceManagerTestFixture, FreeBuffer_InvalidatesHandle ) {
    const auto handle = resource_manager_->create_buffer( {
        .size = 1024,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .name = "TestBuffer",
    } );

    ASSERT_NE( handle.raw, 0 );
    EXPECT_NE( resource_manager_->get_buffer( handle ), VK_NULL_HANDLE );

    resource_manager_->free_buffer( handle );
    EXPECT_EQ( resource_manager_->get_buffer( handle ), VK_NULL_HANDLE );
}

TEST_F( ResourceManagerTestFixture, FreeImage_InvalidatesHandle ) {
    const auto handle = resource_manager_->create_image( {
        .extent = { 1024, 1024, 1 },
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
        .name = "TestImage",
    } );

    ASSERT_NE( handle.raw, 0 );
    EXPECT_NE( resource_manager_->get_image( handle ), VK_NULL_HANDLE );

    resource_manager_->free_image( handle );
    EXPECT_EQ( resource_manager_->get_image( handle ), VK_NULL_HANDLE );
}

//------------------------------------------------------------------------------
// Memory Operations Tests 
//------------------------------------------------------------------------------

TEST_F( ResourceManagerTestFixture, UploadBuffer_HostVisibleMemory ) {
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

TEST_F( ResourceManagerTestFixture, UploadBuffer_HostOnlyMemory ) {
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

TEST_F( ResourceManagerTestFixture, UploadImage_WriteAndReadBack ) {
    // Create test pattern with non-zero data
    std::array<uint8_t, 16 * 16 * 4> test_data{};
    for(size_t i = 0; i < test_data.size(); i++) {
        test_data[i] = static_cast<uint8_t>((i * 7 + 13) % 256); // Generate repeating but non-zero pattern
    }
    std::array<uint8_t, 16 * 16 * 4> read_back_data{};

    constexpr auto data_size = test_data.size();
    const auto image = resource_manager_->create_image( {
        .extent = { 16, 16, 1 },
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .memory_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .name = "TestImage",
    } );

    EXPECT_EQ( resource_manager_->upload_to_image( image, test_data.data(), data_size ), data_size );
    EXPECT_EQ( resource_manager_->read_from_image( image, read_back_data.data(), data_size ), data_size );
    EXPECT_EQ( test_data, read_back_data );
}

//------------------------------------------------------------------------------
// Error Handling & Validation Tests
//------------------------------------------------------------------------------

TEST_F( ResourceManagerTestFixture, UploadBuffer_FailsAfterFree ) {
    constexpr std::array data = { 1, 2, 3, 4, 5, 6, 7, 8 };
    const auto buffer = resource_manager_->create_buffer( {
        .size = 1234,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .memory_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .name = "TestBuffer",
    } );

    resource_manager_->free_buffer( buffer );
    EXPECT_EQ( resource_manager_->upload_to_buffer( buffer, data.data(), data.size() ), 0 );

    // Verify that an error was logged
    const auto& entries = mock_logger_->get_entries();
    EXPECT_FALSE( entries.empty() );

    // Check that at least one entry contains error about invalid buffer handle
    bool found_error = false;
    for ( const auto& [level, message] : entries ) {
        if ( level == aloe::LogLevel::Error && message.find( "Invalid buffer handle" ) != std::string::npos ) {
            found_error = true;
            break;
        }
    }
    EXPECT_TRUE( found_error );
}

TEST_F( ResourceManagerTestFixture, ReadBuffer_FailsAfterFree ) {
    constexpr std::array data = { 1, 2, 3, 4, 5, 6, 7, 8 };
    std::array<int, 8> read_back_data;

    const auto buffer = resource_manager_->create_buffer( {
        .size = data.size() * sizeof( int ),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .memory_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .name = "TestBuffer",
    } );

    resource_manager_->upload_to_buffer( buffer, data.data(), data.size() * sizeof( int ) );

    resource_manager_->free_buffer( buffer );
    EXPECT_EQ( resource_manager_->read_from_buffer( buffer, read_back_data.data(), data.size() * sizeof( int ) ), 0 );

    // Verify that an error was logged
    const auto& entries = mock_logger_->get_entries();
    EXPECT_FALSE( entries.empty() );

    // Check that at least one entry contains error about invalid buffer handle
    bool found_error = false;
    for ( const auto& [level, message] : entries ) {
        if ( level == aloe::LogLevel::Error && message.find( "Invalid buffer handle" ) != std::string::npos ) {
            found_error = true;
            break;
        }
    }
    EXPECT_TRUE( found_error );
}

TEST_F( ResourceManagerTestFixture, UploadImage_FailsAfterFree ) {
    constexpr std::array<uint8_t, 16 * 16 * 4> test_data{};
    const auto image = resource_manager_->create_image( {
        .extent = { 16, 16, 1 },
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .memory_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .name = "TestImage",
    } );

    resource_manager_->free_image( image );
    EXPECT_EQ( resource_manager_->upload_to_image( image, test_data.data(), test_data.size() ), 0 );

    // Verify that an error was logged
    const auto& entries = mock_logger_->get_entries();
    EXPECT_FALSE( entries.empty() );

    bool found_error = false;
    for ( const auto& [level, message] : entries ) {
        if ( level == aloe::LogLevel::Error && message.find( "Invalid image handle" ) != std::string::npos ) {
            found_error = true;
            break;
        }
    }
    EXPECT_TRUE( found_error );
}

TEST_F( ResourceManagerTestFixture, ReadImage_FailsAfterFree ) {
    constexpr std::array<uint8_t, 16 * 16 * 4> test_data{};
    std::array<uint8_t, 16 * 16 * 4> read_back_data{};

    const auto image = resource_manager_->create_image( {
        .extent = { 16, 16, 1 },
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .memory_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .name = "TestImage",
    } );

    resource_manager_->upload_to_image( image, test_data.data(), test_data.size() );

    resource_manager_->free_image( image );
    EXPECT_EQ( resource_manager_->read_from_image( image, read_back_data.data(), test_data.size() ), 0 );

    // Verify that an error was logged
    const auto& entries = mock_logger_->get_entries();
    EXPECT_FALSE( entries.empty() );

    bool found_error = false;
    for ( const auto& [level, message] : entries ) {
        if ( level == aloe::LogLevel::Error && message.find( "Invalid image handle" ) != std::string::npos ) {
            found_error = true;
            break;
        }
    }
    EXPECT_TRUE( found_error );
}

TEST_F( ResourceManagerTestFixture, GetBuffer_ValidatesHandle ) {
    // First create a buffer
    const auto handle = resource_manager_->create_buffer( {
        .size = 1024,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .name = "TestBuffer",
    } );

    ASSERT_NE( handle.raw, 0 );
    EXPECT_NE( resource_manager_->get_buffer( handle ), VK_NULL_HANDLE );

    // Free the buffer then try to access it
    resource_manager_->free_buffer( handle );
    EXPECT_EQ( resource_manager_->get_buffer( handle ), VK_NULL_HANDLE );

    // Verify that an error was logged
    const auto& entries = mock_logger_->get_entries();
    EXPECT_FALSE( entries.empty() );

    // Check that at least one entry contains error about invalid buffer handle
    bool found_error = false;
    for ( const auto& [level, message] : entries ) {
        if ( level == aloe::LogLevel::Error && message.find( "Invalid buffer handle" ) != std::string::npos ) {
            found_error = true;
            break;
        }
    }
    EXPECT_TRUE( found_error );
}

TEST_F( ResourceManagerTestFixture, BufferHandle_ValidatesVersion ) {
    constexpr std::array<int, 4> a_data = { 1, 2, 3, 4 };
    constexpr std::array<int, 4> b_data = { 5, 6, 7, 8 };
    constexpr auto data_size = 4 * sizeof( int );

    const auto handle_a = resource_manager_->create_buffer( {
        .size = 1024,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .memory_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .name = "BufferA",
    } );


    // Upload some test data to buffer A
    EXPECT_EQ( resource_manager_->upload_to_buffer( handle_a, a_data.data(), data_size ), data_size );

    // Free the buffer, verify it goes invalid instantly.
    resource_manager_->free_buffer( handle_a );
    EXPECT_EQ( resource_manager_->get_buffer( handle_a ), VK_NULL_HANDLE );

    // Create a new buffer that should likely reuse the same slot
    const auto handle_b = resource_manager_->create_buffer( {
        .size = 1024,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .memory_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .name = "BufferB",
    } );

    // Upload different test data to buffer B
    EXPECT_EQ( resource_manager_->upload_to_buffer( handle_b, b_data.data(), data_size ), data_size );
    EXPECT_NE( resource_manager_->get_buffer( handle_b ), VK_NULL_HANDLE );
    EXPECT_EQ( resource_manager_->get_buffer( handle_a ), VK_NULL_HANDLE );

    // Read back data from handle_B to verify it's the right data
    std::array<int, 4> read_back_data{};
    EXPECT_EQ( resource_manager_->read_from_buffer( handle_b, read_back_data.data(), data_size ), data_size );
    EXPECT_EQ( read_back_data, b_data );

    // Attempt to read from handle_A (should fail)
    std::array<int, 4> invalid_read_data{};
    EXPECT_EQ( resource_manager_->read_from_buffer( handle_a, invalid_read_data.data(), data_size ), 0 );
}

TEST_F( ResourceManagerTestFixture, ImageHandle_ValidatesVersion ) {
    // Create distinct test patterns for each image
    std::array<uint8_t, 16 * 16 * 4> a_data{};
    std::array<uint8_t, 16 * 16 * 4> b_data{};
    
    // Fill a_data with a pattern
    for(size_t i = 0; i < a_data.size(); i++) {
        a_data[i] = static_cast<uint8_t>((i * 3 + 7) % 256);
    }
    // Fill b_data with a different pattern
    for(size_t i = 0; i < b_data.size(); i++) {
        b_data[i] = static_cast<uint8_t>((i * 5 + 11) % 256);
    }
    
    constexpr auto data_size = 16 * 16 * 4;

    const auto handle_a = resource_manager_->create_image( {
        .extent = { 16, 16, 1 },
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .memory_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .name = "ImageA",
    } );

    // Upload test data to image A
    EXPECT_EQ( resource_manager_->upload_to_image( handle_a, a_data.data(), data_size ), data_size );

    // Free the image, verify it goes invalid instantly
    resource_manager_->free_image( handle_a );
    EXPECT_EQ( resource_manager_->get_image( handle_a ), VK_NULL_HANDLE );

    // Create a new image that should likely reuse the same slot
    const auto handle_b = resource_manager_->create_image( {
        .extent = { 16, 16, 1 },
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .memory_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .name = "ImageB",
    } );

    // Upload different test data to image B
    EXPECT_EQ( resource_manager_->upload_to_image( handle_b, b_data.data(), data_size ), data_size );
    EXPECT_NE( resource_manager_->get_image( handle_b ), VK_NULL_HANDLE );
    EXPECT_EQ( resource_manager_->get_image( handle_a ), VK_NULL_HANDLE );

    // Read back data from handle_B to verify it's the right data
    std::array<uint8_t, 16 * 16 * 4> read_back_data{};
    EXPECT_EQ( resource_manager_->read_from_image( handle_b, read_back_data.data(), data_size ), data_size );
    EXPECT_EQ( read_back_data, b_data );

    // Attempt to read from handle_A (should fail)
    std::array<uint8_t, 16 * 16 * 4> invalid_read_data{};
    EXPECT_EQ( resource_manager_->read_from_image( handle_a, invalid_read_data.data(), data_size ), 0 );
}

//------------------------------------------------------------------------------
// Performance & Stress Tests
//------------------------------------------------------------------------------

TEST_F( ResourceManagerTestFixture, StressTest_MultipleAllocationsAndFrees ) {
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

        ASSERT_NE( handle.raw, 0 );
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

        ASSERT_NE( handle.raw, 0 );
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
