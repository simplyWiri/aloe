#include <aloe/core/Device.h>
#include <aloe/core/PipelineManager.h>
#include <aloe/core/ResourceManager.h>
#include <aloe/util/log.h>

#include <GLFW/glfw3.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <numeric>

#include <spirv-tools/libspirv.hpp>

class PipelineManagerTestFixture : public ::testing::Test {
protected:
    std::shared_ptr<aloe::MockLogger> mock_logger_;
    std::unique_ptr<aloe::Device> device_;
    std::shared_ptr<aloe::PipelineManager> pipeline_manager_;
    std::shared_ptr<aloe::ResourceManager> resource_manager_;
    spvtools::SpirvTools spirv_tools_{ SPV_ENV_VULKAN_1_3 };

    void SetUp() override {
        mock_logger_ = std::make_shared<aloe::MockLogger>();
        aloe::set_logger( mock_logger_ );
        aloe::set_logger_level( aloe::LogLevel::Warn );

        device_ = std::make_unique<aloe::Device>( aloe::DeviceSettings{ .enable_validation = true, .headless = true } );
        resource_manager_ = device_->make_resource_manager();
        pipeline_manager_ = device_->make_pipeline_manager( { "resources" } );

        spirv_tools_.SetMessageConsumer(
            [&]( spv_message_level_t level, const char* source, const spv_position_t& position, const char* message ) {
                EXPECT_TRUE( level > SPV_MSG_WARNING )
                    << source << " errored at line: " << position.line << ", with error: " << message;
            } );
    }

    void TearDown() override {
        resource_manager_.reset();
        pipeline_manager_.reset();
        device_.reset( nullptr );

        auto& debug_info = aloe::Device::debug_info();
        EXPECT_EQ( debug_info.num_warning, 0 );
        EXPECT_EQ( debug_info.num_error, 0 );

        if ( debug_info.num_warning > 0 || debug_info.num_error > 0 ) {
            for ( const auto& [level, message] : mock_logger_->get_entries() ) { std::cerr << message << std::endl; }
        }
    }

    // Helper to compile and validate SPIR-V
    std::expected<aloe::PipelineHandle, std::string> compile_and_validate( const aloe::ComputePipelineInfo& info ) {
        auto result = pipeline_manager_->compile_pipeline( info );
        if ( result ) {
            EXPECT_TRUE( spirv_tools_.Validate( pipeline_manager_->get_pipeline_spirv( *result ) ) )
                << "SPIR-V validation failed for pipeline compiled from: " << info.compute_shader.name;
        }
        return result;
    }

    std::string make_compute_shader( const std::string& body,
                                     const std::string& uniforms = "",
                                     const std::string& entry_point = "compute_main",
                                     uint32_t num_threads_x = 64 ) {
        return std::format( R"(
import aloe;

[shader("compute")]
[numthreads({}, 1, 1)]
void {}(uint3 id : SV_DispatchThreadID{}) {{
    {}
}}
)",
                            num_threads_x,
                            entry_point,
                            uniforms.empty() ? "" : ( ", " + uniforms ),
                            body );
    }


    aloe::BufferHandle create_and_upload_buffer( const char* name, const std::vector<float>& data = {} ) {
        const VkDeviceSize buffer_size = sizeof( float ) * data.size();

        aloe::BufferDesc desc = { .size = buffer_size,
                                  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  .memory_usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .memory_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                      VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                  .name = name };

        aloe::BufferHandle handle = resource_manager_->create_buffer( desc );

        if ( !data.empty() ) {
            const VkDeviceSize uploaded_bytes = resource_manager_->upload_to_buffer( handle, data.data(), buffer_size );
            EXPECT_EQ( uploaded_bytes, buffer_size ) << "Failed to upload buffer: " << name;
        }

        return handle;
    }


    void execute_compute_shader( const std::function<void( VkCommandBuffer )>& record_commands ) {
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        const auto compute_queue = device_->queues_by_capability( VK_QUEUE_COMPUTE_BIT ).front();

        VkCommandPoolCreateInfo pool_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = compute_queue.family_index,
        };
        ASSERT_EQ( vkCreateCommandPool( device_->device(), &pool_info, nullptr, &command_pool ), VK_SUCCESS )
            << "Failed to create command pool.";

        VkCommandBufferAllocateInfo alloc_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        ASSERT_EQ( vkAllocateCommandBuffers( device_->device(), &alloc_info, &command_buffer ), VK_SUCCESS )
            << "Failed to allocate command buffer.";

        VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        ASSERT_EQ( vkBeginCommandBuffer( command_buffer, &begin_info ), VK_SUCCESS )
            << "Failed to begin command buffer.";

        record_commands( command_buffer );

        ASSERT_EQ( vkEndCommandBuffer( command_buffer ), VK_SUCCESS ) << "Failed to end command buffer.";

        VkSubmitInfo submit_info{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffer,
        };

        VkFence fence = VK_NULL_HANDLE;
        VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        ASSERT_EQ( vkCreateFence( device_->device(), &fence_info, nullptr, &fence ), VK_SUCCESS )
            << "Failed to create fence.";

        ASSERT_EQ( vkQueueSubmit( compute_queue.queue, 1, &submit_info, fence ), VK_SUCCESS )
            << "Failed to submit command buffer.";
        ASSERT_EQ( vkWaitForFences( device_->device(), 1, &fence, VK_TRUE, UINT64_MAX ), VK_SUCCESS )
            << "Failed to wait for fence.";

        // Destroy resources.
        vkDestroyFence( device_->device(), fence, nullptr );
        vkDestroyCommandPool( device_->device(), command_pool, nullptr );
    }
};

#define COMPUTE_ENTRY " [shader(\"compute\")] "

//------------------------------------------------------------------------------
// Basic Compilation Tests
//------------------------------------------------------------------------------

TEST_F( PipelineManagerTestFixture, Compile_SimpleShaderFromFile ) {
    const auto shader = aloe::ShaderCompileInfo{ .name = "test.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();
}

TEST_F( PipelineManagerTestFixture, Compile_SimpleShaderFromSource ) {
    pipeline_manager_->set_virtual_file( "virtual_test.slang", COMPUTE_ENTRY "void main() { }" );
    const auto shader = aloe::ShaderCompileInfo{ .name = "virtual_test.slang", .entry_point = "main" };

    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();
}

TEST_F( PipelineManagerTestFixture, Compile_ShaderWithDefines ) {
    const auto shader = aloe::ShaderCompileInfo{ .name = "define_shader.slang", .entry_point = "main" };
    pipeline_manager_->set_virtual_file( "define_shader.slang", COMPUTE_ENTRY "void main() { int x = MY_DEFINE; }" );
    pipeline_manager_->set_define( "MY_DEFINE", "1" );

    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();
}

TEST_F( PipelineManagerTestFixture, Compile_FailsOnInvalidShader ) {
    pipeline_manager_->set_virtual_file( "virtual_test.slang", COMPUTE_ENTRY "void main(" );
    const auto shader = aloe::ShaderCompileInfo{ .name = "virtual_test.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );

    ASSERT_FALSE( handle.has_value() );
    EXPECT_TRUE( handle.error().find( "Failed to compile shader" ) != std::string::npos );
    EXPECT_TRUE( handle.error().find( "virtual_test.slang" ) != std::string::npos );
}

//------------------------------------------------------------------------------
// Resource Binding and Validation Tests
//------------------------------------------------------------------------------

TEST_F( PipelineManagerTestFixture, Binding_InvalidResourceReturnsFalse ) {
    pipeline_manager_->set_virtual_file(
        "invalid_resource.slang",
        make_compute_shader( "", "uniform aloe::BufferHandle buf, uniform aloe::ImageHandle img", "main", 1 ) );

    const auto shader = aloe::ShaderCompileInfo{ .name = "invalid_resource.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    // Test invalid buffer handle
    auto buf_uniform = pipeline_manager_->get_uniform_handle<aloe::BufferHandle>( *handle, "buf" );
    const auto fake_buffer = aloe::BufferHandle( 999 );
    EXPECT_FALSE( pipeline_manager_->set_uniform( buf_uniform.set_value( fake_buffer ),
                                                  aloe::usage( fake_buffer, aloe::ComputeStorageRead ) ) );

    // Test invalid image handle
    auto img_uniform = pipeline_manager_->get_uniform_handle<aloe::ImageHandle>( *handle, "img" );
    const auto fake_image = aloe::ImageHandle( 888 );
    EXPECT_FALSE( pipeline_manager_->set_uniform( img_uniform.set_value( fake_image ),
                                                  aloe::usage( fake_image, aloe::ComputeStorageRead ) ) );
}

TEST_F( PipelineManagerTestFixture, Binding_FreedResourceErrorsOnBind ) {
    pipeline_manager_->set_virtual_file( "freed_resource.slang",
                                         make_compute_shader( "", "uniform aloe::BufferHandle buf", "main", 1 ) );

    const auto shader = aloe::ShaderCompileInfo{ .name = "freed_resource.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    // Create and set a valid buffer
    auto buffer = create_and_upload_buffer( "FreedBuffer", { 1.0f, 2.0f, 3.0f } );
    auto buf_uniform = pipeline_manager_->get_uniform_handle<aloe::BufferHandle>( *handle, "buf" );
    EXPECT_TRUE( pipeline_manager_->set_uniform( buf_uniform.set_value( buffer ),
                                                 aloe::usage( buffer, aloe::ComputeStorageRead ) ) );

    // Free the buffer before bind_pipeline
    resource_manager_->free_buffer( buffer );

    // Execute should generate error but not crash
    execute_compute_shader(
        [&]( VkCommandBuffer cmd ) { EXPECT_FALSE( pipeline_manager_->bind_pipeline( *handle, cmd ) ); } );
}

TEST_F( PipelineManagerTestFixture, Binding_MultipleResourcesSameSlot ) {
    pipeline_manager_->set_virtual_file( "multi_resource.slang",
                                         make_compute_shader( "", "uniform aloe::BufferHandle buf", "main", 1 ) );

    const auto shader = aloe::ShaderCompileInfo{ .name = "multi_resource.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    // Create two buffers
    auto buffer1 = create_and_upload_buffer( "Buffer1", { 1.0f, 2.0f, 3.0f } );
    auto buffer2 = create_and_upload_buffer( "Buffer2", { 4.0f, 5.0f, 6.0f } );

    auto buf_uniform = pipeline_manager_->get_uniform_handle<aloe::BufferHandle>( *handle, "buf" );

    // First binding should succeed
    EXPECT_TRUE( pipeline_manager_->set_uniform( buf_uniform.set_value( buffer1 ),
                                                 aloe::usage( buffer1, aloe::ComputeStorageRead ) ) );

    pipeline_manager_->bind_slots();
    resource_manager_->free_buffer( buffer1 );

    // Second binding to same slot should succeed and replace first binding
    EXPECT_TRUE( pipeline_manager_->set_uniform( buf_uniform.set_value( buffer2 ),
                                                 aloe::usage( buffer2, aloe::ComputeStorageRead ) ) );
}

TEST_F( PipelineManagerTestFixture, Binding_ResourceVersionValidation ) {
    pipeline_manager_->set_virtual_file( "version_test.slang",
                                         make_compute_shader( "", "uniform aloe::BufferHandle buf", "main", 1 ) );

    const auto shader = aloe::ShaderCompileInfo{ .name = "version_test.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    // Create initial buffer and bind it
    auto buffer = create_and_upload_buffer( "VersionBuffer", { 1.0f, 2.0f, 3.0f } );
    auto buf_uniform = pipeline_manager_->get_uniform_handle<aloe::BufferHandle>( *handle, "buf" );
    EXPECT_TRUE( pipeline_manager_->set_uniform( buf_uniform.set_value( buffer ),
                                                 aloe::usage( buffer, aloe::ComputeStorageRead ) ) );
    pipeline_manager_->bind_slots();

    // Free the buffer
    resource_manager_->free_buffer( buffer );

    // Create a new buffer - it might get the same ID but should have a different version
    auto new_buffer = create_and_upload_buffer( "NewVersionBuffer", { 4.0f, 5.0f, 6.0f } );

    // Execute should fail or generate error about invalid resource version
    execute_compute_shader(
        [&]( VkCommandBuffer cmd ) { EXPECT_FALSE( pipeline_manager_->bind_pipeline( *handle, cmd ) ); } );

    // Binding the new buffer should succeed
    EXPECT_TRUE( pipeline_manager_->set_uniform( buf_uniform.set_value( new_buffer ),
                                                 aloe::usage( new_buffer, aloe::ComputeStorageRead ) ) );

    pipeline_manager_->bind_slots();

    execute_compute_shader(
        [&]( VkCommandBuffer cmd ) { EXPECT_TRUE( pipeline_manager_->bind_pipeline( *handle, cmd ) ); } );
}

//------------------------------------------------------------------------------
// Dependency Tracking Tests
//------------------------------------------------------------------------------

TEST_F( PipelineManagerTestFixture, Dependency_VirtualFileBasic ) {
    pipeline_manager_->set_virtual_file( "test.slang",
                                         "module test; public int add(int a, int b) { return 5 + a + b; }" );
    pipeline_manager_->set_virtual_file( "main_shader.slang",
                                         "import test;" COMPUTE_ENTRY " void main() { int x = add(1, 2); }" );

    const aloe::ShaderCompileInfo shader{ .name = "main_shader.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();
}

TEST_F( PipelineManagerTestFixture, Dependency_VirtualFileTriggersRecompile ) {
    pipeline_manager_->set_virtual_file( "test.slang",
                                         "module test; public int add(int a, int b) { return 5 + a + b; }" );
    pipeline_manager_->set_virtual_file( "main_shader.slang",
                                         "import test;" COMPUTE_ENTRY "void main() { int x = add(1, 2); }" );

    const aloe::ShaderCompileInfo shader{ .name = "main_shader.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    uint64_t baseline_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( baseline_version, 1 );

    pipeline_manager_->set_virtual_file( "test.slang",
                                         "module test; public int add(int a, int b) { return 8 + a + b; }" );

    const uint64_t next_version = pipeline_manager_->get_pipeline_version( *handle );

    EXPECT_NE( baseline_version, next_version );
    EXPECT_EQ( next_version, baseline_version + 1 );
}

TEST_F( PipelineManagerTestFixture, Dependency_UnrelatedFileNoRecompile ) {
    pipeline_manager_->set_virtual_file( "test.slang",
                                         "module test; public int add(int a, int b) { return 5 + a + b; }" );
    pipeline_manager_->set_virtual_file( "main_shader.slang", COMPUTE_ENTRY "void main() { int x = 5; }" );

    const aloe::ShaderCompileInfo shader{ .name = "main_shader.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    uint64_t baseline_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( baseline_version, 1 );

    pipeline_manager_->set_virtual_file( "test.slang",
                                         "module test; public int add(int a, int b) { return 8 + a + b; }" );

    const uint64_t next_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( baseline_version, next_version );
}

TEST_F( PipelineManagerTestFixture, Dependency_UpdateRecompilesShader ) {
    pipeline_manager_->set_virtual_file( "mid.slang", "module mid; public int square(int x) { return x * x; }" );
    pipeline_manager_->set_virtual_file( "main.slang",
                                         "import mid;" COMPUTE_ENTRY "void main() { int x = square(4); }" );

    const aloe::ShaderCompileInfo shader{ .name = "main.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    const uint64_t baseline_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( baseline_version, 1 );

    pipeline_manager_->set_virtual_file( "mid.slang", "module mid; public int square(int x) { return x * x * x; }" );

    const uint64_t new_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( new_version, baseline_version + 1 );
}

TEST_F( PipelineManagerTestFixture, Dependency_TransitiveUpdateRecompiles ) {
    pipeline_manager_->set_virtual_file( "common.slang",
                                         "module common; public int add(int a, int b) { return a + b; }" );
    pipeline_manager_->set_virtual_file(
        "mid.slang",
        "module mid; import common; public int triple(int x) { return add(x, add(x, x)); }" );
    pipeline_manager_->set_virtual_file( "main.slang",
                                         "import mid;" COMPUTE_ENTRY "void main() { int x = triple(3); }" );

    const aloe::ShaderCompileInfo shader{ .name = "main.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    const uint64_t initial_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( initial_version, 1 );

    pipeline_manager_->set_virtual_file( "common.slang",
                                         "module common; public int add(int a, int b) { return 1 + a + b; }" );

    const uint64_t new_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( new_version, initial_version + 1 );
}

TEST_F( PipelineManagerTestFixture, Dependency_DiamondDependencyRecompilesOnce ) {
    /*
        Dependency graph:
             main.slang (A)
            /           \
       mid_left        mid_right
           \           /
             shared_dep (D)
    */
    pipeline_manager_->set_virtual_file( "shared_dep.slang", "module shared_dep; public int val() { return 42; }" );
    pipeline_manager_->set_virtual_file( "mid_left.slang",
                                         "import shared_dep; module mid_left; public int left() { return val(); }" );
    pipeline_manager_->set_virtual_file( "mid_right.slang",
                                         "import shared_dep; module mid_right; public int right() { return val(); }" );
    pipeline_manager_->set_virtual_file( "main.slang",
                                         "import mid_left;import mid_right;" COMPUTE_ENTRY
                                         "void main() { int x = left() + right(); }" );

    const aloe::ShaderCompileInfo shader{ .name = "main.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    const uint64_t version_before = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( version_before, 1 );

    pipeline_manager_->set_virtual_file( "shared_dep.slang", "module shared_dep; public int val() { return 1337; }" );

    const uint64_t version_after = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( version_after, version_before + 1 );
}

//------------------------------------------------------------------------------
// Uniform Block Tests
//------------------------------------------------------------------------------

TEST_F( PipelineManagerTestFixture, Uniform_BasicCompute ) {
    pipeline_manager_->set_virtual_file(
        "basic_uniform.slang",
        make_compute_shader(
            R"(
                RWByteAddressBuffer buf = outbuf_handle.get();
                if (id.x == 0) {
                    buf.Store<float>(0, time);
                    buf.Store<int>(sizeof(uint), frameCount);
                }
            )",
            "uniform float time, uniform int frameCount, uniform aloe::BufferHandle outbuf_handle",
            "compute_main",
            1 ) );

    const aloe::ShaderCompileInfo shader_info{ .name = "basic_uniform.slang", .entry_point = "compute_main" };
    const auto pipeline_handle = compile_and_validate( { shader_info } );
    ASSERT_TRUE( pipeline_handle ) << pipeline_handle.error();
    auto h_time = pipeline_manager_->get_uniform_handle<float>( *pipeline_handle, "time" );
    auto h_frame = pipeline_manager_->get_uniform_handle<int>( *pipeline_handle, "frameCount" );
    auto h_outbuf = pipeline_manager_->get_uniform_handle<aloe::BufferHandle>( *pipeline_handle, "outbuf_handle" );
    auto outbuf = create_and_upload_buffer( "UniformOut", { 0.0f, 0.0f } );

    pipeline_manager_->set_uniform( h_time.set_value( 123.45f ) );
    pipeline_manager_->set_uniform( h_frame.set_value( 99 ) );
    EXPECT_TRUE( pipeline_manager_->set_uniform( h_outbuf.set_value( outbuf ),
                                                 aloe::usage( outbuf, aloe::ComputeStorageWrite ) ) );

    pipeline_manager_->bind_slots();

    execute_compute_shader( [&]( VkCommandBuffer cmd ) {
        pipeline_manager_->bind_pipeline( *pipeline_handle, cmd );
        vkCmdDispatch( cmd, 1, 1, 1 );
    } );

    std::vector<uint32_t> result_data( 2 );
    resource_manager_->read_from_buffer( outbuf, result_data.data(), sizeof( uint32_t ) * 2 );
    EXPECT_FLOAT_EQ( std::bit_cast<float>( result_data[0] ), 123.45f );
    EXPECT_EQ( result_data[1], 99u );
}

TEST_F( PipelineManagerTestFixture, Uniform_StructType ) {
    struct MyParams {
        float intensity;
        int mode;
    };
    pipeline_manager_->set_virtual_file( "struct_uniform.slang",
                                         std::string( "struct MyParams { float intensity; int mode; };\n" ) +
                                             make_compute_shader(
                                                 R"(
                RWByteAddressBuffer buf = outbuf_handle.get();
                buf.Store<float>(0, params.intensity);
                buf.Store<int>(sizeof(float), params.mode);
            )",
                                                 "uniform MyParams params, uniform aloe::BufferHandle outbuf_handle",
                                                 "compute_main",
                                                 1 ) );
    const aloe::ShaderCompileInfo shader_info{ .name = "struct_uniform.slang", .entry_point = "compute_main" };
    const auto pipeline_handle = compile_and_validate( { shader_info } );
    ASSERT_TRUE( pipeline_handle ) << pipeline_handle.error();
    auto h_params = pipeline_manager_->get_uniform_handle<MyParams>( *pipeline_handle, "params" );
    auto h_outbuf = pipeline_manager_->get_uniform_handle<aloe::BufferHandle>( *pipeline_handle, "outbuf_handle" );
    auto outbuf = create_and_upload_buffer( "StructUniformOut", { 0.0f, 0.0f } );

    MyParams test_params = { 0.75f, 2 };
    pipeline_manager_->set_uniform( h_params.set_value( test_params ) );
    pipeline_manager_->set_uniform( h_outbuf.set_value( outbuf ), aloe::usage( outbuf, aloe::ComputeStorageWrite ) );
    pipeline_manager_->bind_slots();

    execute_compute_shader( [&]( VkCommandBuffer cmd ) {
        pipeline_manager_->bind_pipeline( *pipeline_handle, cmd );
        vkCmdDispatch( cmd, 1, 1, 1 );
    } );

    std::vector<uint32_t> result_data( 2 );
    resource_manager_->read_from_buffer( outbuf, result_data.data(), sizeof( uint32_t ) * 2 );
    EXPECT_FLOAT_EQ( std::bit_cast<float>( result_data[0] ), 0.75f );
    EXPECT_EQ( result_data[1], 2u );
}

TEST_F( PipelineManagerTestFixture, Uniform_PersistenceAcrossDispatches ) {
    pipeline_manager_->set_virtual_file( "persist_uniform.slang",
                                         make_compute_shader(
                                             R"(
                RWByteAddressBuffer buf = outbuf_handle.get();
                buf.Store<float>(0, myval);
            )",
                                             "uniform float myval, uniform aloe::BufferHandle outbuf_handle",
                                             "compute_main",
                                             1 ) );
    const aloe::ShaderCompileInfo shader_info{ .name = "persist_uniform.slang", .entry_point = "compute_main" };
    const auto pipeline_handle = compile_and_validate( { shader_info } );
    ASSERT_TRUE( pipeline_handle ) << pipeline_handle.error();
    auto h_myval = pipeline_manager_->get_uniform_handle<float>( *pipeline_handle, "myval" );
    auto h_outbuf = pipeline_manager_->get_uniform_handle<aloe::BufferHandle>( *pipeline_handle, "outbuf_handle" );
    auto outbuf = create_and_upload_buffer( "PersistUniformOut", { 0.0f } );

    pipeline_manager_->set_uniform( h_myval.set_value( 1.5f ) );
    pipeline_manager_->set_uniform( h_outbuf.set_value( outbuf ), aloe::usage( outbuf, aloe::ComputeStorageWrite ) );
    pipeline_manager_->bind_slots();

    execute_compute_shader( [&]( VkCommandBuffer cmd ) {
        pipeline_manager_->bind_pipeline( *pipeline_handle, cmd );
        vkCmdDispatch( cmd, 1, 1, 1 );
    } );

    std::vector<uint32_t> result_data( 1 );
    resource_manager_->read_from_buffer( outbuf, result_data.data(), sizeof( uint32_t ) );
    EXPECT_FLOAT_EQ( std::bit_cast<float>( result_data[0] ), 1.5f );

    pipeline_manager_->set_uniform( h_myval.set_value( 7.25f ) );
    execute_compute_shader( [&]( VkCommandBuffer cmd ) {
        pipeline_manager_->bind_pipeline( *pipeline_handle, cmd );
        vkCmdDispatch( cmd, 1, 1, 1 );
    } );
    resource_manager_->read_from_buffer( outbuf, result_data.data(), sizeof( uint32_t ) );
    EXPECT_FLOAT_EQ( std::bit_cast<float>( result_data[0] ), 7.25f );
}

TEST_F( PipelineManagerTestFixture, Uniform_AliasedTypesAtSameOffset ) {
    pipeline_manager_->set_virtual_file( "aliased_uniform.slang",
                                         R"(
        [shader("vertex")]
        void vertex_main(uniform float param_one) { }
        [shader("fragment")]
        void fragment_main(uniform int param_one) { }
        )" );
    aloe::GraphicsPipelineInfo pipeline_info{
        .vertex_shader = { .name = "aliased_uniform.slang", .entry_point = "vertex_main" },
        .fragment_shader = { .name = "aliased_uniform.slang", .entry_point = "fragment_main" }
    };
    const auto pipeline_handle = pipeline_manager_->compile_pipeline( pipeline_info );
    ASSERT_FALSE( pipeline_handle.has_value() );
    EXPECT_TRUE( pipeline_handle.error().find( "param_one" ) != std::string::npos ||
                 pipeline_handle.error().find( "float" ) != std::string::npos ||
                 pipeline_handle.error().find( "int" ) != std::string::npos );
}

TEST_F( PipelineManagerTestFixture, Uniform_OverlappingRangesDifferentNames ) {
    pipeline_manager_->set_virtual_file( "overlap_uniform.slang",
                                         R"(
        [shader("vertex")]
        void vertex_main(uniform float foo) { }
        [shader("fragment")]
        void fragment_main(uniform float bar) { }
        )" );
    aloe::GraphicsPipelineInfo pipeline_info{
        .vertex_shader = { .name = "overlap_uniform.slang", .entry_point = "vertex_main" },
        .fragment_shader = { .name = "overlap_uniform.slang", .entry_point = "fragment_main" }
    };
    const auto pipeline_handle = pipeline_manager_->compile_pipeline( pipeline_info );
    ASSERT_FALSE( pipeline_handle.has_value() );
    EXPECT_TRUE( pipeline_handle.error().find( "overlap" ) != std::string::npos ||
                 pipeline_handle.error().find( "conflict" ) != std::string::npos );
}

TEST_F( PipelineManagerTestFixture, Uniform_SupersetRange ) {
    pipeline_manager_->set_virtual_file( "overlap_uniform.slang",
                                         R"(
        [shader("vertex")]
        void vertex_main(uniform float foo) { }
        [shader("fragment")]
        void fragment_main(uniform float foo, uniform float frag_only) { }
        )" );
    aloe::GraphicsPipelineInfo pipeline_info{
        .vertex_shader = { .name = "overlap_uniform.slang", .entry_point = "vertex_main" },
        .fragment_shader = { .name = "overlap_uniform.slang", .entry_point = "fragment_main" }
    };
    const auto pipeline_handle = pipeline_manager_->compile_pipeline( pipeline_info );
    ASSERT_TRUE( pipeline_handle.has_value() );
}

//------------------------------------------------------------------------------
// Multi-Entry Point and File Organization Tests
//------------------------------------------------------------------------------

TEST_F( PipelineManagerTestFixture, MultiEntry_SingleFile ) {
    pipeline_manager_->set_virtual_file( "multi_entry.slang",
                                         R"(
        struct VertexOutput {
            float4 position : SV_Position;
            float4 color : COLOR;
        };

        [shader("vertex")]
        VertexOutput vertex_main(uint vertex_id : SV_VertexID) {
            VertexOutput output;
            const float2 positions[] = {
                float2(-1, -1), float2(3, -1), float2(-1, 3)
            };
            output.position = float4(positions[vertex_id], 0, 1);
            output.color = float4(1, 0, 0, 1);
            return output;
        }

        [shader("fragment")]
        float4 fragment_main(VertexOutput input) : SV_Target {
            return input.color;
        }
        )" );

    aloe::GraphicsPipelineInfo pipeline_info{
        .vertex_shader = { .name = "multi_entry.slang", .entry_point = "vertex_main" },
        .fragment_shader = { .name = "multi_entry.slang", .entry_point = "fragment_main" }
    };

    const auto handle = pipeline_manager_->compile_pipeline( pipeline_info );
    ASSERT_TRUE( handle ) << handle.error();

    const auto initial_version = pipeline_manager_->get_pipeline_version( *handle );

    pipeline_manager_->set_virtual_file( "multi_entry.slang",
                                         R"(
        struct VertexOutput {
            float4 position : SV_Position;
            float4 color : COLOR;
        };

        [shader("vertex")]
        VertexOutput vertex_main(uint vertex_id : SV_VertexID) {
            VertexOutput output;
            const float2 positions[] = {
                float2(-1, -1), float2(3, -1), float2(-1, 3)
            };
            output.position = float4(positions[vertex_id], 0, 1);
            output.color = float4(0, 1, 0, 1);
            return output;
        }

        [shader("fragment")]
        float4 fragment_main(VertexOutput input) : SV_Target {
            return input.color;
        }
        )" );

    const auto updated_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_GT( updated_version, initial_version );
}

TEST_F( PipelineManagerTestFixture, MultiEntry_SeparateFilesSharedDependency ) {
    pipeline_manager_->set_virtual_file( "shared.slang",
                                         R"(
        struct VertexOutput {
            float4 position : SV_Position;
            float4 color : COLOR;
        };
        )" );

    pipeline_manager_->set_virtual_file( "vertex.slang",
                                         R"(
        #include "shared.slang"

        [shader("vertex")]
        VertexOutput main(uint vertex_id : SV_VertexID) {
            VertexOutput output;
            const float2 positions[] = {
                float2(-1, -1), float2(3, -1), float2(-1, 3)
            };
            output.position = float4(positions[vertex_id], 0, 1);
            output.color = float4(1, 0, 0, 1);
            return output;
        }
        )" );

    pipeline_manager_->set_virtual_file( "fragment.slang",
                                         R"(
        #include "shared.slang"

        [shader("fragment")]
        float4 main(VertexOutput input) : SV_Target {
            return input.color;
        }
        )" );

    aloe::GraphicsPipelineInfo pipeline_info{ .vertex_shader = { .name = "vertex.slang", .entry_point = "main" },
                                              .fragment_shader = { .name = "fragment.slang", .entry_point = "main" } };

    const auto handle = pipeline_manager_->compile_pipeline( pipeline_info );
    ASSERT_TRUE( handle ) << handle.error();

    const auto initial_version = pipeline_manager_->get_pipeline_version( *handle );

    pipeline_manager_->set_virtual_file( "shared.slang",
                                         R"(
        struct VertexOutput {
            float4 position : SV_Position;
            float4 color : COLOR;
            float2 uv : TEXCOORD;
        };
        )" );

    const auto updated_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_GT( updated_version, initial_version );
}

TEST_F( PipelineManagerTestFixture, MultiEntry_DifferentPipelineHandles ) {
    pipeline_manager_->set_virtual_file( "multi_entry.slang",
                                         COMPUTE_ENTRY "void main1() { int a = 1; }" COMPUTE_ENTRY
                                                       "void main2() { int b = 2; }" );

    const aloe::ShaderCompileInfo shader1{ .name = "multi_entry.slang", .entry_point = "main1" };
    const aloe::ShaderCompileInfo shader2{ .name = "multi_entry.slang", .entry_point = "main2" };

    const auto handle1 = compile_and_validate( { shader1 } );
    const auto handle2 = compile_and_validate( { shader2 } );

    ASSERT_TRUE( handle1.has_value() ) << handle1.error();
    ASSERT_TRUE( handle2.has_value() ) << handle2.error();

    EXPECT_NE( handle1->id, handle2->id );

    const auto first_spirv = pipeline_manager_->get_pipeline_spirv( *handle1 );
    const auto second_spirv = pipeline_manager_->get_pipeline_spirv( *handle2 );

    EXPECT_TRUE( spirv_tools_.Validate( first_spirv ) );
    EXPECT_TRUE( spirv_tools_.Validate( second_spirv ) );

    EXPECT_NE( first_spirv, second_spirv );
}

//------------------------------------------------------------------------------
// End-to-End Tests
//------------------------------------------------------------------------------

TEST_F( PipelineManagerTestFixture, E2E_BufferDataModification ) {
    constexpr size_t num_elements = 64;
    constexpr VkDeviceSize buffer_size = num_elements * sizeof( float );

    std::vector<float> initial_data( num_elements );
    std::iota( initial_data.begin(), initial_data.end(), 1.0f );

    std::vector<float> expected_data = initial_data;
    for ( float& val : expected_data ) { val *= 2.0f; }

    auto buffer_handle = create_and_upload_buffer( "DataModificationBuffer", initial_data );

    std::string shader_body = R"(
        RWByteAddressBuffer buf = data_buffer.get();
        uint address = id.x * sizeof(float);

        float value = buf.Load<float>(address);
        buf.Store<float>(address, value * 2.0f);
    )";

    pipeline_manager_->set_virtual_file(
        "e2e_compute.slang",
        make_compute_shader( shader_body, "uniform aloe::BufferHandle data_buffer", "compute_main", num_elements ) );

    aloe::ShaderCompileInfo shader_info{ .name = "e2e_compute.slang", .entry_point = "compute_main" };
    auto pipeline_handle_result = compile_and_validate( { shader_info } );
    ASSERT_TRUE( pipeline_handle_result.has_value() ) << pipeline_handle_result.error();
    aloe::PipelineHandle pipeline_handle = *pipeline_handle_result;

    auto h_data_buffer = pipeline_manager_->get_uniform_handle<aloe::BufferHandle>( pipeline_handle, "data_buffer" );

    pipeline_manager_->set_uniform( h_data_buffer.set_value( buffer_handle ),
                                    aloe::usage( buffer_handle, aloe::ComputeSampledRead ) );
    pipeline_manager_->bind_slots();

    execute_compute_shader( [&]( VkCommandBuffer cmd ) {
        pipeline_manager_->bind_pipeline( pipeline_handle, cmd );
        vkCmdDispatch( cmd, 1, 1, 1 );
    } );

    std::vector<float> readback_data( num_elements );
    VkDeviceSize read_bytes = resource_manager_->read_from_buffer( buffer_handle, readback_data.data(), buffer_size );
    ASSERT_EQ( read_bytes, buffer_size );

    ASSERT_EQ( readback_data.size(), expected_data.size() );
    for ( size_t i = 0; i < num_elements; ++i ) {
        EXPECT_FLOAT_EQ( readback_data[i], expected_data[i] ) << "Mismatch at index " << i;
    }
}

TEST_F( PipelineManagerTestFixture, E2E_ThreeBufferElementWiseMultiply ) {
    constexpr size_t num_elements = 64;

    std::vector<float> input_data_1( num_elements );
    std::vector<float> input_data_2( num_elements );
    std::vector<float> expected_data( num_elements );

    std::iota( input_data_1.begin(), input_data_1.end(), 1.0f );
    std::iota( input_data_2.begin(), input_data_2.end(), 2.0f );
    for ( size_t i = 0; i < num_elements; ++i ) { expected_data[i] = input_data_1[i] * input_data_2[i]; }

    auto buffer1 = create_and_upload_buffer( "InputBuffer1", input_data_1 );
    auto buffer2 = create_and_upload_buffer( "InputBuffer2", input_data_2 );
    auto buffer3 = create_and_upload_buffer( "OutputBuffer", std::vector<float>( num_elements ) );

    std::string shader_body = R"(
        RWByteAddressBuffer buffer1 = buffer1_handle.get();
        RWByteAddressBuffer buffer2 = buffer2_handle.get();
        RWByteAddressBuffer buffer3 = buffer3_handle.get();

        uint index = id.x * sizeof(float);
        float value1 = buffer1.Load<float>(index);
        float value2 = buffer2.Load<float>(index);

        buffer3.Store<float>(index, value1 * value2);
    )";

    pipeline_manager_->set_virtual_file( "multiply_buffers.slang",
                                         make_compute_shader( shader_body,
                                                              "uniform aloe::BufferHandle buffer1_handle, "
                                                              "uniform aloe::BufferHandle buffer2_handle, "
                                                              "uniform aloe::BufferHandle buffer3_handle",
                                                              "compute_main",
                                                              num_elements ) );

    const aloe::ShaderCompileInfo shader_info{ .name = "multiply_buffers.slang", .entry_point = "compute_main" };
    aloe::PipelineHandle pipeline_handle = *compile_and_validate( { shader_info } );

    auto h_buffer1 = pipeline_manager_->get_uniform_handle<aloe::BufferHandle>( pipeline_handle, "buffer1_handle" );
    auto h_buffer2 = pipeline_manager_->get_uniform_handle<aloe::BufferHandle>( pipeline_handle, "buffer2_handle" );
    auto h_buffer3 = pipeline_manager_->get_uniform_handle<aloe::BufferHandle>( pipeline_handle, "buffer3_handle" );

    const auto read_1 = aloe::usage( buffer1, aloe::ComputeStorageRead );
    const auto read_2 = aloe::usage( buffer2, aloe::ComputeStorageRead );
    const auto write_1 = aloe::usage( buffer3, aloe::ComputeStorageWrite );

    ASSERT_TRUE( pipeline_manager_->set_uniform( h_buffer1.set_value( buffer1 ), read_1 ) );
    ASSERT_TRUE( pipeline_manager_->set_uniform( h_buffer2.set_value( buffer2 ), read_2 ) );
    ASSERT_TRUE( pipeline_manager_->set_uniform( h_buffer3.set_value( buffer3 ), write_1 ) );

    pipeline_manager_->bind_slots();

    execute_compute_shader( [&]( VkCommandBuffer cmd ) {
        pipeline_manager_->bind_pipeline( pipeline_handle, cmd );
        vkCmdDispatch( cmd, 1, 1, 1 );
    } );

    std::vector<float> readback_data( num_elements );
    VkDeviceSize read_bytes =
        resource_manager_->read_from_buffer( buffer3, readback_data.data(), sizeof( float ) * num_elements );
    ASSERT_EQ( read_bytes, sizeof( float ) * num_elements );

    ASSERT_EQ( readback_data.size(), expected_data.size() );
    for ( size_t i = 0; i < num_elements; ++i ) {
        EXPECT_FLOAT_EQ( readback_data[i], expected_data[i] ) << "Mismatch at index " << i;
    }
}

TEST_F( PipelineManagerTestFixture, E2E_ImageProcedural ) {
    constexpr uint32_t image_size = 8;
    constexpr uint32_t total_pixels = image_size * image_size;
    constexpr uint32_t grid_size = 2;

    auto image = resource_manager_->create_image( {
        .extent = { image_size, image_size, 1 },
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .tiling = VK_IMAGE_TILING_LINEAR,
        .name = "GridPatternImage",
    } );

    std::string shader_body = R"(
        RWTexture2D<float4> output_tex = output_image.get();

        bool is_white = ((id.x / 2 + id.y / 2) % 2) == 0;
        float4 color = is_white ? float4(1.0, 1.0, 1.0, 1.0)
                                : float4(0.0, 0.0, 0.0, 1.0);

        output_tex[id.xy] = color;
    )";

    pipeline_manager_->set_virtual_file(
        "grid_pattern.slang",
        make_compute_shader( shader_body, "uniform aloe::ImageHandle output_image", "compute_main", image_size ) );

    const aloe::ShaderCompileInfo shader_info{ .name = "grid_pattern.slang", .entry_point = "compute_main" };
    auto pipeline_handle_result = compile_and_validate( { shader_info } );
    ASSERT_TRUE( pipeline_handle_result.has_value() ) << pipeline_handle_result.error();
    aloe::PipelineHandle pipeline_handle = *pipeline_handle_result;

    auto uni_output_image = pipeline_manager_->get_uniform_handle<aloe::ImageHandle>( pipeline_handle, "output_image" );
    pipeline_manager_->set_uniform( uni_output_image.set_value( image ),
                                    aloe::usage( image, aloe::ComputeStorageWrite ) );
    pipeline_manager_->bind_slots();

    execute_compute_shader( [&]( VkCommandBuffer cmd ) {
        VkImageMemoryBarrier2KHR barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR,
                                          .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                                          .srcAccessMask = VK_ACCESS_2_NONE,
                                          .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                          .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                          .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                          .image = resource_manager_->get_image( image ),
                                          .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                                .baseMipLevel = 0,
                                                                .levelCount = 1,
                                                                .baseArrayLayer = 0,
                                                                .layerCount = 1 } };

        VkDependencyInfo dependency_info{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                          .imageMemoryBarrierCount = 1,
                                          .pImageMemoryBarriers = &barrier };

        vkCmdPipelineBarrier2KHR( cmd, &dependency_info );

        pipeline_manager_->bind_pipeline( pipeline_handle, cmd );
        vkCmdDispatch( cmd, 1, image_size, 1 );

        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

        vkCmdPipelineBarrier2KHR( cmd, &dependency_info );
    } );

    std::vector<float> readback_data( total_pixels * 4 );
    VkDeviceSize read_bytes =
        resource_manager_->read_from_image( image, readback_data.data(), readback_data.size() * sizeof( float ) );
    ASSERT_EQ( read_bytes, readback_data.size() * sizeof( float ) );

    for ( uint32_t y = 0; y < image_size; y++ ) {
        for ( uint32_t x = 0; x < image_size; x++ ) {
            uint32_t pixel_idx = ( y * image_size + x ) * 4;

            bool should_be_white = ( ( x / grid_size + y / grid_size ) % 2 ) == 0;
            float expected = should_be_white ? 1.0f : 0.0f;

            EXPECT_FLOAT_EQ( readback_data[pixel_idx + 0], expected )
                << "Mismatch at pixel (" << x << "," << y << ") red component";
            EXPECT_FLOAT_EQ( readback_data[pixel_idx + 1], expected )
                << "Mismatch at pixel (" << x << "," << y << ") green component";
            EXPECT_FLOAT_EQ( readback_data[pixel_idx + 2], expected )
                << "Mismatch at pixel (" << x << "," << y << ") blue component";
            EXPECT_FLOAT_EQ( readback_data[pixel_idx + 3], 1.0f )
                << "Mismatch at pixel (" << x << "," << y << ") alpha component";
        }
    }
}
