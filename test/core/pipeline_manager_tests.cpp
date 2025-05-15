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
        pipeline_manager_ = device_->make_pipeline_manager( { "resources" } );
        resource_manager_ = device_->make_resource_manager();

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

// Compile a simple shader from file and verify success and valid output
TEST_F( PipelineManagerTestFixture, SimpleShaderCompilationFromFile ) {
    const auto shader = aloe::ShaderCompileInfo{ .name = "test.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();
}

TEST_F( PipelineManagerTestFixture, SimpleShaderCompilationFromSource ) {
    pipeline_manager_->set_virtual_file( "virtual_test.slang", COMPUTE_ENTRY "void main() { }" );
    const auto shader = aloe::ShaderCompileInfo{ .name = "virtual_test.slang", .entry_point = "main" };

    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();
}

// Compilation should fail for an invalid shader path
TEST_F( PipelineManagerTestFixture, SimpleShaderErrorFails ) {
    // Syntactical error in compilation
    pipeline_manager_->set_virtual_file( "virtual_test.slang", COMPUTE_ENTRY "void main(" );
    const auto shader = aloe::ShaderCompileInfo{ .name = "virtual_test.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );

    ASSERT_FALSE( handle.has_value() );
    EXPECT_TRUE( handle.error().find( "Failed to compile shader" ) != std::string::npos );
    EXPECT_TRUE( handle.error().find( "virtual_test.slang" ) != std::string::npos );
}

// Compiling a shader with defines should work
TEST_F( PipelineManagerTestFixture, CompileShaderWithDefines ) {
    const auto shader = aloe::ShaderCompileInfo{ .name = "define_shader.slang", .entry_point = "main" };
    pipeline_manager_->set_virtual_file( "define_shader.slang", COMPUTE_ENTRY "void main() { int x = MY_DEFINE; }" );
    pipeline_manager_->set_define( "MY_DEFINE", "1" );

    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();
}

// Setting a new define should trigger recompilation with different output
TEST_F( PipelineManagerTestFixture, ShaderRecompilesWithDefine ) {
    const auto shader = aloe::ShaderCompileInfo{ .name = "define_shader.slang", .entry_point = "main" };
    pipeline_manager_->set_virtual_file( "define_shader.slang", COMPUTE_ENTRY "void main() { int x = MY_DEFINE; }" );
    pipeline_manager_->set_define( "MY_DEFINE", "1" );
    const auto handle = compile_and_validate( { shader } );

    const auto initial_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( initial_version, 1 );

    pipeline_manager_->set_define( "MY_DEFINE", "2" );
    const auto recompiled_version = pipeline_manager_->get_pipeline_version( *handle );

    EXPECT_NE( initial_version, recompiled_version );
    EXPECT_EQ( recompiled_version, initial_version + 1 );
}

// Recompiling after changing a dependent virtual file should increment version
TEST_F( PipelineManagerTestFixture, VirtualFileDependencyWorks ) {
    // Set up the virtual include file & a shader that dependents this file
    pipeline_manager_->set_virtual_file( "test.slang",
                                         "module test; public int add(int a, int b) { return 5 + a + b; }" );
    pipeline_manager_->set_virtual_file( "main_shader.slang",
                                         "import test;" COMPUTE_ENTRY " void main() { int x = add(1, 2); }" );

    const aloe::ShaderCompileInfo shader{ .name = "main_shader.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();
}

TEST_F( PipelineManagerTestFixture, VirtualFileDependencyTriggersRecompilation ) {
    // Set up the virtual include file & a shader that dependents this file
    pipeline_manager_->set_virtual_file( "test.slang",
                                         "module test; public int add(int a, int b) { return 5 + a + b; }" );
    pipeline_manager_->set_virtual_file( "main_shader.slang",
                                         "import test;" COMPUTE_ENTRY "void main() { int x = add(1, 2); }" );

    const aloe::ShaderCompileInfo shader{ .name = "main_shader.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    uint64_t baseline_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( baseline_version, 1 );

    // Modify the shared include, this should automatically result in the pipeline being recompiled immediately
    pipeline_manager_->set_virtual_file( "test.slang",
                                         "module test; public int add(int a, int b) { return 8 + a + b; }" );

    const uint64_t next_version = pipeline_manager_->get_pipeline_version( *handle );

    EXPECT_NE( baseline_version, next_version );
    EXPECT_EQ( next_version, baseline_version + 1 );
}

// Changing an unrelated virtual file should not trigger recompilation
TEST_F( PipelineManagerTestFixture, UnrelatedVirtualFileDependencyDoesntTriggerUpdate ) {
    // Set up the virtual include file & a shader that dependents this file
    pipeline_manager_->set_virtual_file( "test.slang",
                                         "module test; public int add(int a, int b) { return 5 + a + b; }" );
    pipeline_manager_->set_virtual_file( "main_shader.slang", COMPUTE_ENTRY "void main() { int x = 5; }" );

    const aloe::ShaderCompileInfo shader{ .name = "main_shader.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    uint64_t baseline_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( baseline_version, 1 );

    // Modify the shared include, this should not result in the pipeline being recompiled - because `main_shader.slang`
    // does not depend on `test` at all.
    pipeline_manager_->set_virtual_file( "test.slang",
                                         "module test; public int add(int a, int b) { return 8 + a + b; }" );

    const uint64_t next_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( baseline_version, next_version );
}

// Updating a dependent file causes the pipeline to automatically recompile
TEST_F( PipelineManagerTestFixture, DependencyUpdateRecompilesFinalShader ) {
    // Set up shader module graph: `main` depends on `mid`
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

// Changing a transitive dependency should recompile all dependents
TEST_F( PipelineManagerTestFixture, TransitiveDependencyUpdateRecompilesFinalShader ) {
    // main -> mid -> common
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

    // Modify the transitive dependency: common.slang
    pipeline_manager_->set_virtual_file( "common.slang",
                                         "module common; public int add(int a, int b) { return 1 + a + b; }" );

    const uint64_t new_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( new_version, initial_version + 1 );
}

// A diamond dependency graph should trigger only one recompilation of the top-level shader
TEST_F( PipelineManagerTestFixture, DiamondDependencyRecompilesOnce ) {
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

    // Modify shared leaf dependency
    pipeline_manager_->set_virtual_file( "shared_dep.slang", "module shared_dep; public int val() { return 1337; }" );

    const uint64_t version_after = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( version_after, version_before + 1 );
}

// Compiling the same shader with different entry points should produce different pipeline handles
TEST_F( PipelineManagerTestFixture, MultipleEntryPointsProduceDifferentPipelines ) {
    pipeline_manager_->set_virtual_file( "multi_entry.slang",
                                         COMPUTE_ENTRY "void main1() { int a = 1; }" COMPUTE_ENTRY
                                                       "void main2() { int b = 2; }" );

    const aloe::ShaderCompileInfo shader1{ .name = "multi_entry.slang", .entry_point = "main1" };
    const aloe::ShaderCompileInfo shader2{ .name = "multi_entry.slang", .entry_point = "main2" };

    const auto handle1 = compile_and_validate( { shader1 } );
    const auto handle2 = compile_and_validate( { shader2 } );

    ASSERT_TRUE( handle1.has_value() ) << handle1.error();
    ASSERT_TRUE( handle2.has_value() ) << handle2.error();

    // They should be different handles
    EXPECT_NE( handle1->id, handle2->id );

    // Validate both SPIR-V blobs
    const auto first_spirv = pipeline_manager_->get_pipeline_spirv( *handle1 );
    const auto second_spirv = pipeline_manager_->get_pipeline_spirv( *handle2 );

    EXPECT_TRUE( spirv_tools_.Validate( first_spirv ) );
    EXPECT_TRUE( spirv_tools_.Validate( second_spirv ) );

    EXPECT_NE( first_spirv, second_spirv );
}

// Modifying a shared include should cause all entry points that depend on it to be recompiled
TEST_F( PipelineManagerTestFixture, SharedIncludeRecompilesAllEntryPoints ) {
    pipeline_manager_->set_virtual_file( "shared.slang", "module shared; public int val() { return 1; }" );

    pipeline_manager_->set_virtual_file( "multi_entry.slang",
                                         "import shared;" COMPUTE_ENTRY "void main1() { int a = val(); }" COMPUTE_ENTRY
                                         "void main2() { int b = val(); }" );

    const aloe::ShaderCompileInfo shader1{ .name = "multi_entry.slang", .entry_point = "main1" };
    const aloe::ShaderCompileInfo shader2{ .name = "multi_entry.slang", .entry_point = "main2" };

    const auto handle1 = compile_and_validate( { shader1 } );
    const auto handle2 = compile_and_validate( { shader2 } );

    ASSERT_TRUE( handle1.has_value() ) << handle1.error();
    ASSERT_TRUE( handle2.has_value() ) << handle2.error();

    const uint64_t version1_before = pipeline_manager_->get_pipeline_version( *handle1 );
    const uint64_t version2_before = pipeline_manager_->get_pipeline_version( *handle2 );
    EXPECT_EQ( version1_before, 1 );
    EXPECT_EQ( version2_before, 1 );

    // Modify the shared include
    pipeline_manager_->set_virtual_file( "shared.slang", "module shared; public int val() { return 999; }" );

    const uint64_t version1_after = pipeline_manager_->get_pipeline_version( *handle1 );
    const uint64_t version2_after = pipeline_manager_->get_pipeline_version( *handle2 );

    EXPECT_EQ( version1_after, version1_before + 1 );
    EXPECT_EQ( version2_after, version2_before + 1 );
}

// Circular virtual file dependency should be detected
// Circular virtual file dependencies should not cause infinite loops or crashes during compilation
TEST_F( PipelineManagerTestFixture, CircularIncludesHandledGracefully ) {
    pipeline_manager_->set_virtual_file( "a.slang", "import b;" COMPUTE_ENTRY "void main() { }" );
    pipeline_manager_->set_virtual_file( "b.slang", "import c;" );
    pipeline_manager_->set_virtual_file( "c.slang", "import a;" );// Circular: a -> b -> c -> a

    const aloe::ShaderCompileInfo shader{ .name = "a.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );

    ASSERT_FALSE( handle.has_value() );
    EXPECT_TRUE( handle.error().find( "circular" ) != std::string::npos ||
                 handle.error().find( "cycle" ) != std::string::npos ||
                 handle.error().find( "import" ) != std::string::npos );
}


TEST_F( PipelineManagerTestFixture, UniformBlockBasicCompute ) {
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
    pipeline_manager_->bind_buffer( *resource_manager_, outbuf );

    // --- Set Uniforms ---
    pipeline_manager_->set_uniform( *pipeline_handle, h_time.set_value( 123.45f ) );
    pipeline_manager_->set_uniform( *pipeline_handle, h_frame.set_value( 99 ) );
    pipeline_manager_->set_uniform( *pipeline_handle, h_outbuf.set_value( outbuf ) );

    // --- Dispatch ---
    execute_compute_shader( [&]( VkCommandBuffer cmd ) {
        pipeline_manager_->bind_pipeline( *pipeline_handle, cmd );
        vkCmdDispatch( cmd, 1, 1, 1 );
    } );

    // --- Verify ---
    std::vector<uint32_t> result_data( 2 );
    resource_manager_->read_from_buffer( outbuf, result_data.data(), sizeof( uint32_t ) * 2 );
    EXPECT_FLOAT_EQ( std::bit_cast<float>( result_data[0] ), 123.45f );
    EXPECT_EQ( result_data[1], 99u );
}

TEST_F( PipelineManagerTestFixture, UniformBlockStruct ) {
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
    pipeline_manager_->bind_buffer( *resource_manager_, outbuf );

    // --- Set Uniforms ---
    MyParams test_params = { 0.75f, 2 };
    pipeline_manager_->set_uniform( *pipeline_handle, h_params.set_value( test_params ) );
    pipeline_manager_->set_uniform( *pipeline_handle, h_outbuf.set_value( outbuf ) );

    // --- Dispatch ---
    execute_compute_shader( [&]( VkCommandBuffer cmd ) {
        pipeline_manager_->bind_pipeline( *pipeline_handle, cmd );
        vkCmdDispatch( cmd, 1, 1, 1 );
    } );

    // --- Verify ---
    std::vector<uint32_t> result_data( 2 );
    resource_manager_->read_from_buffer( outbuf, result_data.data(), sizeof( uint32_t ) * 2 );
    EXPECT_FLOAT_EQ( std::bit_cast<float>( result_data[0] ), 0.75f );
    EXPECT_EQ( result_data[1], 2u );
}

TEST_F( PipelineManagerTestFixture, UniformPersistenceAcrossDispatches ) {
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
    pipeline_manager_->bind_buffer( *resource_manager_, outbuf );

    // --- First dispatch ---
    pipeline_manager_->set_uniform( *pipeline_handle, h_myval.set_value( 1.5f ) );
    pipeline_manager_->set_uniform( *pipeline_handle, h_outbuf.set_value( outbuf ) );
    execute_compute_shader( [&]( VkCommandBuffer cmd ) {
        pipeline_manager_->bind_pipeline( *pipeline_handle, cmd );
        vkCmdDispatch( cmd, 1, 1, 1 );
    } );
    std::vector<uint32_t> result_data( 1 );
    resource_manager_->read_from_buffer( outbuf, result_data.data(), sizeof( uint32_t ) );
    EXPECT_FLOAT_EQ( std::bit_cast<float>( result_data[0] ), 1.5f );

    // --- Second dispatch ---
    pipeline_manager_->set_uniform( *pipeline_handle, h_myval.set_value( 7.25f ) );
    execute_compute_shader( [&]( VkCommandBuffer cmd ) {
        pipeline_manager_->bind_pipeline( *pipeline_handle, cmd );
        vkCmdDispatch( cmd, 1, 1, 1 );
    } );
    resource_manager_->read_from_buffer( outbuf, result_data.data(), sizeof( uint32_t ) );
    EXPECT_FLOAT_EQ( std::bit_cast<float>( result_data[0] ), 7.25f );
}

// Compile a simple shader from file and verify success and valid output
TEST_F( PipelineManagerTestFixture, SimpleBufferBindingWorks ) {
    // Compile the test shader, ensure it is valid.
    const auto shader = aloe::ShaderCompileInfo{ .name = "test.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    // Allocate a buffer
    const auto buffer = resource_manager_->create_buffer( {
        .size = sizeof( float ) * 128,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .name = "Test Buffer",
    } );

    const auto result = pipeline_manager_->bind_buffer( *resource_manager_, buffer );
    EXPECT_EQ( result, VK_SUCCESS );
}

TEST_F( PipelineManagerTestFixture, InvalidBufferBindingFailsGracefully ) {
    // Compile the test shader, ensure it is valid.
    const auto shader = aloe::ShaderCompileInfo{ .name = "test.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    // Allocate a fake buffer, which does not exist.
    const auto buffer = aloe::BufferHandle( 1, 5, 23 );
    const auto result = pipeline_manager_->bind_buffer( *resource_manager_, buffer );
    EXPECT_NE( result, VK_SUCCESS );
}

// E2E Tests execute compute shaders on the GPU to verify behaviour.
TEST_F( PipelineManagerTestFixture, E2E_BufferDataModification ) {
    constexpr size_t num_elements = 64;
    constexpr VkDeviceSize buffer_size = num_elements * sizeof( float );

    std::vector<float> initial_data( num_elements );
    std::iota( initial_data.begin(), initial_data.end(), 1.0f );

    std::vector<float> expected_data = initial_data;
    for ( float& val : expected_data ) { val *= 2.0f; }

    auto buffer_handle = create_and_upload_buffer( "DataModificationBuffer", initial_data );
    pipeline_manager_->bind_buffer( *resource_manager_, buffer_handle );

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

    pipeline_manager_->set_uniform( pipeline_handle, h_data_buffer.set_value( buffer_handle ) );

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

    pipeline_manager_->bind_buffer( *resource_manager_, buffer1 );
    pipeline_manager_->bind_buffer( *resource_manager_, buffer2 );
    pipeline_manager_->bind_buffer( *resource_manager_, buffer3 );

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

    pipeline_manager_->set_uniform( pipeline_handle, h_buffer1.set_value( buffer1 ) );
    pipeline_manager_->set_uniform( pipeline_handle, h_buffer2.set_value( buffer2 ) );
    pipeline_manager_->set_uniform( pipeline_handle, h_buffer3.set_value( buffer3 ) );

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

TEST_F(PipelineManagerTestFixture, UniformBlockAliasedTypesAtSameOffset) {
    pipeline_manager_->set_virtual_file(
        "aliased_uniform.slang",
        R"(
        // Both uniforms at offset 0, different types, same name
        [shader("vertex")]
        void vertex_main(uniform float param_one) { }
        [shader("fragment")]
        void fragment_main(uniform int param_one) { }
        )"
    );
    aloe::GraphicsPipelineInfo pipeline_info{
        .vertex_shader = { .name = "aliased_uniform.slang", .entry_point = "vertex_main" },
        .fragment_shader = { .name = "aliased_uniform.slang", .entry_point = "fragment_main" }
    };
    const auto pipeline_handle = pipeline_manager_->compile_pipeline(pipeline_info);
    ASSERT_FALSE(pipeline_handle.has_value());
    EXPECT_TRUE(pipeline_handle.error().find("param_one") != std::string::npos ||
                pipeline_handle.error().find("float") != std::string::npos ||
                pipeline_handle.error().find("int") != std::string::npos);
}

TEST_F(PipelineManagerTestFixture, UniformBlockOverlappingRangesDifferentNames) {
    pipeline_manager_->set_virtual_file(
        "overlap_uniform.slang",
        R"(
        // Both uniforms at offset 0, different names, same types
        [shader("vertex")]
        void vertex_main(uniform float foo) { }
        [shader("fragment")]
        void fragment_main(uniform float bar) { }
        )"
    );
    aloe::GraphicsPipelineInfo pipeline_info{
        .vertex_shader = { .name = "overlap_uniform.slang", .entry_point = "vertex_main" },
        .fragment_shader = { .name = "overlap_uniform.slang", .entry_point = "fragment_main" }
    };
    const auto pipeline_handle = pipeline_manager_->compile_pipeline(pipeline_info);
    ASSERT_FALSE(pipeline_handle.has_value());
    EXPECT_TRUE(pipeline_handle.error().find("overlap") != std::string::npos ||
                pipeline_handle.error().find("conflict") != std::string::npos);
}

TEST_F(PipelineManagerTestFixture, UniformBlockSupersetRange) {
    pipeline_manager_->set_virtual_file(
        "overlap_uniform.slang",
        R"(
        [shader("vertex")]
        void vertex_main(uniform float foo) { }
        [shader("fragment")]
        void fragment_main(uniform float foo, uniform float frag_only) { }
        )"
    );
    aloe::GraphicsPipelineInfo pipeline_info{
        .vertex_shader = { .name = "overlap_uniform.slang", .entry_point = "vertex_main" },
        .fragment_shader = { .name = "overlap_uniform.slang", .entry_point = "fragment_main" }
    };
    const auto pipeline_handle = pipeline_manager_->compile_pipeline(pipeline_info);
    ASSERT_TRUE(pipeline_handle.has_value());
}


TEST_F(PipelineManagerTestFixture, MultipleEntryPointsSingleFile) {
    pipeline_manager_->set_virtual_file(
        "multi_entry.slang",
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
        )"
    );

    aloe::GraphicsPipelineInfo pipeline_info{
        .vertex_shader = { .name = "multi_entry.slang", .entry_point = "vertex_main" },
        .fragment_shader = { .name = "multi_entry.slang", .entry_point = "fragment_main" }
    };

    const auto handle = pipeline_manager_->compile_pipeline(pipeline_info);
    ASSERT_TRUE(handle) << handle.error();

    // Store initial versions
    const auto initial_version = pipeline_manager_->get_pipeline_version(*handle);

    // Modify the shader and verify versions are updated
    pipeline_manager_->set_virtual_file(
        "multi_entry.slang",
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
            output.color = float4(0, 1, 0, 1);  // Changed from red to green
            return output;
        }

        [shader("fragment")]
        float4 fragment_main(VertexOutput input) : SV_Target {
            return input.color;
        }
        )"
    );

    const auto updated_version = pipeline_manager_->get_pipeline_version(*handle);
    EXPECT_GT(updated_version, initial_version);
}

TEST_F(PipelineManagerTestFixture, SeparateFilesWithSharedDependency) {
    // First create a shared header file
    pipeline_manager_->set_virtual_file(
        "shared.slang",
        R"(
        struct VertexOutput {
            float4 position : SV_Position;
            float4 color : COLOR;
        };
        )"
    );

    // Create vertex shader
    pipeline_manager_->set_virtual_file(
        "vertex.slang",
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
        )"
    );

    // Create fragment shader
    pipeline_manager_->set_virtual_file(
        "fragment.slang",
        R"(
        #include "shared.slang"

        [shader("fragment")]
        float4 main(VertexOutput input) : SV_Target {
            return input.color;
        }
        )"
    );

    aloe::GraphicsPipelineInfo pipeline_info{
        .vertex_shader = { .name = "vertex.slang", .entry_point = "main" },
        .fragment_shader = { .name = "fragment.slang", .entry_point = "main" }
    };

    const auto handle = pipeline_manager_->compile_pipeline(pipeline_info);
    ASSERT_TRUE(handle) << handle.error();

    // Store initial versions
    const auto initial_version = pipeline_manager_->get_pipeline_version(*handle);

    // Modify the shared header and verify both shaders are recompiled
    pipeline_manager_->set_virtual_file(
        "shared.slang",
        R"(
        struct VertexOutput {
            float4 position : SV_Position;
            float4 color : COLOR;
            float2 uv : TEXCOORD;  // Added new field
        };
        )"
    );

    const auto updated_version = pipeline_manager_->get_pipeline_version(*handle);
    EXPECT_GT(updated_version, initial_version);
}


