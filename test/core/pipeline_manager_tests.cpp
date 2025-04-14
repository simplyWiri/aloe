#include <aloe/core/PipelineManager.h>
#include <aloe/core/ResourceManager.h>
#include <aloe/util/log.h>

#include <GLFW/glfw3.h>
#include <gtest/gtest.h>

#include <filesystem>

#include <spirv-tools/libspirv.hpp>

class PipelineManagerTestFixture : public ::testing::Test {
protected:
    std::shared_ptr<aloe::MockLogger> mock_logger_;
    std::unique_ptr<aloe::Device> device_;
    std::shared_ptr<aloe::PipelineManager> pipeline_manager_;
    spvtools::SpirvTools spirv_tools_{ SPV_ENV_VULKAN_1_3 };

    void SetUp() override {
        mock_logger_ = std::make_shared<aloe::MockLogger>();
        aloe::set_logger( mock_logger_ );
        aloe::set_logger_level( aloe::LogLevel::Warn );

        device_ = std::make_unique<aloe::Device>( aloe::DeviceSettings{ .enable_validation = true, .headless = true } );
        pipeline_manager_ =
            std::make_shared<aloe::PipelineManager>( *device_, std::vector<std::string>{ "resources" } );

        spirv_tools_.SetMessageConsumer(
            [&]( spv_message_level_t level, const char* source, const spv_position_t& position, const char* message ) {
                EXPECT_TRUE( level > SPV_MSG_WARNING )
                    << source << " errored at line: " << position.line << ", with error: " << message;
            } );
    }

    void TearDown() override {
        pipeline_manager_.reset();
        device_.reset( nullptr );

        auto& debug_info = aloe::Device::debug_info();
        EXPECT_EQ( debug_info.num_warning, 0 );
        EXPECT_EQ( debug_info.num_error, 0 );

        if ( debug_info.num_warning > 0 || debug_info.num_error > 0 ) {
            for ( const auto& [level, message] : mock_logger_->get_entries() ) { std::cerr << message << std::endl; }
        }
    }
};

#define COMPUTE_ENTRY "import aloe; [shader(\"compute\")] "

// Compile a simple shader from file and verify success and valid output
TEST_F( PipelineManagerTestFixture, SimpleShaderCompilationFromFile ) {
    const auto shader = aloe::ShaderCompileInfo{ .name = "test.slang", .entry_point = "main" };
    const auto result = pipeline_manager_->compile_pipeline( { shader } );

    ASSERT_TRUE( result.has_value() ) << result.error();
    EXPECT_TRUE( spirv_tools_.Validate( pipeline_manager_->get_pipeline_spirv( *result ) ) );
}

TEST_F( PipelineManagerTestFixture, SimpleShaderCompilationFromSource ) {
    pipeline_manager_->set_virtual_file( "virtual_test.slang", COMPUTE_ENTRY "void main() { }" );
    const auto shader = aloe::ShaderCompileInfo{ .name = "virtual_test.slang", .entry_point = "main" };
    const auto result = pipeline_manager_->compile_pipeline( { shader } );

    ASSERT_TRUE( result.has_value() ) << result.error();
    EXPECT_TRUE( spirv_tools_.Validate( pipeline_manager_->get_pipeline_spirv( *result ) ) );
}

// Compilation should fail for an invalid shader path
TEST_F( PipelineManagerTestFixture, SimpleShaderErrorFails ) {
    // Syntactical error in compilation
    pipeline_manager_->set_virtual_file( "virtual_test.slang", COMPUTE_ENTRY "void main(" );
    const auto shader = aloe::ShaderCompileInfo{ .name = "virtual_test.slang", .entry_point = "main" };
    const auto result = pipeline_manager_->compile_pipeline( { shader } );

    ASSERT_FALSE( result.has_value() );
    EXPECT_TRUE( result.error().find( "Failed to compile shader" ) != std::string::npos );
    EXPECT_TRUE( result.error().find( "virtual_test.slang" ) != std::string::npos );
}

// Compiling a shader with defines should work
TEST_F( PipelineManagerTestFixture, CompileShaderWithDefines ) {
    const auto shader = aloe::ShaderCompileInfo{ .name = "define_shader.slang", .entry_point = "main" };
    pipeline_manager_->set_virtual_file( "define_shader.slang", COMPUTE_ENTRY "void main() { int x = MY_DEFINE; }" );
    pipeline_manager_->set_define( "MY_DEFINE", "1" );

    const auto handle = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();
}

// Setting a new define should trigger recompilation with different output
TEST_F( PipelineManagerTestFixture, ShaderRecompilesWithDefine ) {
    const auto shader = aloe::ShaderCompileInfo{ .name = "define_shader.slang", .entry_point = "main" };
    pipeline_manager_->set_virtual_file( "define_shader.slang", COMPUTE_ENTRY "void main() { int x = MY_DEFINE; }" );
    pipeline_manager_->set_define( "MY_DEFINE", "1" );
    const auto handle = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );

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
    const auto handle = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();
}

TEST_F( PipelineManagerTestFixture, VirtualFileDependencyTriggersRecompilation ) {
    // Set up the virtual include file & a shader that dependents this file
    pipeline_manager_->set_virtual_file( "test.slang",
                                         "module test; public int add(int a, int b) { return 5 + a + b; }" );
    pipeline_manager_->set_virtual_file( "main_shader.slang",
                                         "import test;" COMPUTE_ENTRY "void main() { int x = add(1, 2); }" );

    const aloe::ShaderCompileInfo shader{ .name = "main_shader.slang", .entry_point = "main" };
    const auto handle = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
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
    const auto handle = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
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
    const auto handle = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
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
    const auto handle = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
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
    const auto handle = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
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

    const auto handle1 = pipeline_manager_->compile_pipeline( { .compute_shader = shader1 } );
    const auto handle2 = pipeline_manager_->compile_pipeline( { .compute_shader = shader2 } );

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

    const auto handle1 = pipeline_manager_->compile_pipeline( { .compute_shader = shader1 } );
    const auto handle2 = pipeline_manager_->compile_pipeline( { .compute_shader = shader2 } );

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
    const auto result = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );

    ASSERT_FALSE( result.has_value() );
    EXPECT_TRUE( result.error().find( "circular" ) != std::string::npos ||
                 result.error().find( "cycle" ) != std::string::npos ||
                 result.error().find( "import" ) != std::string::npos );
}

TEST_F( PipelineManagerTestFixture, CreatesPipelineLayoutFromReflection ) {
    pipeline_manager_->set_virtual_file( "test.slang", R"(
import aloe;

[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID,
                  uniform aloe::BufferHandle buffer,
                  uniform float time) {
    buffer.get<float>()[id.x] = time;
}
)" );

    const aloe::ShaderCompileInfo shader{ .name = "test.slang", .entry_point = "compute_main" };
    const auto result = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
    EXPECT_TRUE( result );

    auto pc_buffer = pipeline_manager_->get_uniform_handle<aloe::BufferHandle>( *result, VK_SHADER_STAGE_COMPUTE_BIT, "buffer" );
    auto pc_time = pipeline_manager_->get_uniform_handle<float>( *result, VK_SHADER_STAGE_COMPUTE_BIT, "time" );

    pc_buffer.set_value( {5} );
    pc_time.set_value( 123.456f );

    // auto uniform_block = pipeline_manager_->get_uniform_block( *result, VK_SHADER_STAGE_COMPUTE_BIT );
    aloe::UniformBlock uniform_block(12);
    uniform_block.set( pc_buffer );
    uniform_block.set( pc_time );

    const auto* data = static_cast<const uint8_t*>(uniform_block.data());
    EXPECT_EQ( *reinterpret_cast<const aloe::BufferHandle*>(data + pc_buffer.offset), aloe::BufferHandle{5} );
    EXPECT_EQ( *reinterpret_cast<const float*>(data + pc_time.offset), 123.456f );
}

TEST_F( PipelineManagerTestFixture, Something) {
    pipeline_manager_->set_virtual_file( "test.slang", R"(
import aloe;

[[vk::push_constant]]
struct VertexPushConstants {
    [[vk::offset(0)]] float4x4 modelMatrix;
};

[shader("vertex")]
void vertex_main(
    [[vk::location(0)]] float3 inPosition : POSITION,
    uniform VertexPushConstants pc,
    out float4 outPosition : SV_Position)
{
    outPosition = mul(pc.modelMatrix, float4(inPosition, 1.0));
};

// Fragment Shader
[[vk::push_constant]]
struct FragmentPushConstants {
    [[vk::offset(64)]] float4 color;
};

[shader("fragment")]
void fragment_main(
    in float4 inPosition : SV_Position,
    uniform FragmentPushConstants pc,
    out float4 outColor : SV_Target)
{
    outColor = pc.color;
};
)" );

    const aloe::ShaderCompileInfo shader{ .name = "test.slang", .entry_point = "compute_main" };
    const auto result = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
    EXPECT_TRUE( result );
}

// TEST_F( PipelineManagerTestFixture, CopiesBufferWithBindlessHandle ) {
//     pipeline_manager_->set_virtual_file( "copy_buffer.slang", R"(
// import aloe;
//
// [shader("compute")]
// [numthreads(64, 1, 1)]
// void compute_main(uint3 id : SV_DispatchThreadID,
//                   uniform aloe::BufferHandle src,
//                   uniform aloe::BufferHandle dst) {
//     dst.get<float>()[id.x] = src.get<float>()[id.x];
// }
// )" );
//
//     const size_t num_elements = 64;
//     std::vector src_data( num_elements, 42.0f );
//     std::vector dst_data( num_elements, 0.0f );
//
//     auto src_buf = device_->resource_manager().create_host_buffer( src_data );
//     auto dst_buf = device_->resource_manager().create_host_buffer( dst_data );
//
//     aloe::ShaderCompileInfo shader{ .name = "copy_buffer.slang", .entry_point = "compute_main" };
//
//     auto result = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
//     ASSERT_TRUE( result );
//
//     pipeline_manager_->bind_global_buffer( src_buf.handle(), 0 );
//     pipeline_manager_->bind_global_buffer( dst_buf.handle(), 1 );
//     pipeline_manager_->dispatch( shader, num_elements );
//
//     auto readback = dst_buf.readback<float>();
//     EXPECT_EQ( readback, src_data );
// }
//
// TEST_F( PipelineManagerTestFixture, FailsGracefullyOnOutOfBoundsHandle ) {
//     pipeline_manager_->set_virtual_file( "out_of_bounds.slang", R"(
// import aloe;
//
// [shader("compute")]
// [numthreads(1, 1, 1)]
// void compute_main(uint3 id : SV_DispatchThreadID,
//                   uniform aloe::BufferHandle bogus) {
//     bogus.get<int>()[0] = 1337;  // Index is intentionally out-of-bounds
// }
// )" );
//
//     aloe::ShaderCompileInfo shader{ .name = "out_of_bounds.slang", .entry_point = "compute_main" };
//
//     auto result = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
//     ASSERT_TRUE( result );
//
//     aloe::BufferHandle invalid_handle{ .id = 9999 };               // Clearly invalid
//     pipeline_manager_->bind_global_buffer( {}, invalid_handle.id );// No actual buffer
//
//     // Expect a validation error from Vulkan layer or graceful fallback in logs
//     pipeline_manager_->dispatch( shader, 1 );
//     EXPECT_GE( aloe::Device::debug_info().num_error, 1 );
// }
//
// TEST_F( PipelineManagerTestFixture, ReportsShaderErrorWhenBindingMissing ) {
//     pipeline_manager_->set_virtual_file( "missing_binding.slang", R"(
// [shader("compute")]
// [numthreads(1, 1, 1)]
// void compute_main(uint3 id : SV_DispatchThreadID) {
//     // Does not import aloe; will miss g_buffers
// }
// )" );
//
//     aloe::ShaderCompileInfo shader{ .name = "missing_binding.slang", .entry_point = "compute_main" };
//
//     auto result = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
//     EXPECT_FALSE( result );// Should fail due to missing aloe import / g_buffers
// }
//
// TEST_F( PipelineManagerTestFixture, PushConstantAccessSucceedsWhenInRange ) {
//     pipeline_manager_->set_virtual_file( "push_constant_ok.slang", R"(
// import aloe;
//
// [shader("compute")]
// [numthreads(1, 1, 1)]
// void compute_main(uint3 id : SV_DispatchThreadID,
//                   uniform aloe::BufferHandle buffer,
//                   uniform float value) {
//     buffer.get<float>()[0] = value;
// }
// )" );
//
//     aloe::ShaderCompileInfo shader{ .name = "push_constant_ok.slang", .entry_point = "compute_main" };
//
//     auto result = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
//     ASSERT_TRUE( result );
//
//     auto buffer = device_->resource_manager().create_host_buffer<float>( { 0.0f } );
//     pipeline_manager_->bind_global_buffer( buffer.handle(), 0 );
//     pipeline_manager_->dispatch( shader, 1, 1, 1, 123.456f );
//
//     auto out = buffer.readback<float>();
//     EXPECT_FLOAT_EQ( out[0], 123.456f );
// }
//
// TEST_F( PipelineManagerTestFixture, HandlesDescriptorOverflowAtRuntime ) {
//     pipeline_manager_->set_virtual_file( "noop.slang", R"(
// import aloe;
//
// [shader("compute")]
// [numthreads(1, 1, 1)]
// void compute_main(uint3 id : SV_DispatchThreadID) {}
// )" );
//
//     aloe::ShaderCompileInfo shader{ .name = "noop.slang", .entry_point = "compute_main" };
//
//     auto result = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
//     ASSERT_TRUE( result );
//
//     // Exceed descriptor limit (arbitrarily use 2048 handles)
//     for ( int i = 0; i < 2048; ++i ) {
//         auto buf = device_->resource_manager().create_host_buffer<float>( { float( i ) } );
//         pipeline_manager_->bind_global_buffer( buf.handle(), i );
//     }
//
//     pipeline_manager_->dispatch( shader, 1 );
//
//     // Expect at least a warning or a Vulkan validation error
//     EXPECT_GE( aloe::Device::debug_info().num_warning + aloe::Device::debug_info().num_error, 1 );
// }
