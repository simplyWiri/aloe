#include <aloe/core/Device.h>
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
                                     const std::string& entry_point = "compute_main" ) {
        return std::format( R"(
[shader("compute")]
[numthreads(1, 1, 1)]
void {}(uint3 id : SV_DispatchThreadID{}) {{
    {}
}}
)",
                            entry_point,
                            uniforms.empty() ? "" : ( ", " + uniforms ),
                            body );
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
        make_compute_shader( "// body irrelevant", "uniform float time, uniform int frameCount" ) );

    const aloe::ShaderCompileInfo shader{ .name = "basic_uniform.slang", .entry_point = "compute_main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle ) << handle.error();

    // 1. Get Handles and check offsets (Slang determines actual offsets)
    auto h_time = pipeline_manager_->get_uniform_handle<float>( *handle, VK_SHADER_STAGE_COMPUTE_BIT, "time" );
    auto h_frame = pipeline_manager_->get_uniform_handle<int>( *handle, VK_SHADER_STAGE_COMPUTE_BIT, "frameCount" );

    // Ensure offsets are distinct and reasonable (depends on packing rules)
    EXPECT_NE( h_time.offset, h_frame.offset );
    EXPECT_LT( h_time.offset, 16 );// Likely within first few bytes
    EXPECT_LT( h_frame.offset, 16 );

    // 2. Get the UniformBlock
    auto& block = pipeline_manager_->get_uniform_block( *handle );

    // 3. Verify Block Size and Bindings
    // Size depends on packing, expect at least sizeof(float) + sizeof(int) = 8
    // Slang might align frameCount to 4 bytes after time, so expect 8.
    const uint32_t expected_size = h_frame.offset + sizeof( int );// Assumes frameCount is last
    ASSERT_EQ( block.size(), expected_size );
    ASSERT_EQ( block.get_bindings().size(), 1 );// One binding for compute stage

    const auto& binding = block.get_bindings()[0];
    EXPECT_EQ( binding.stage_flags, VK_SHADER_STAGE_COMPUTE_BIT );
    EXPECT_EQ( binding.offset, h_time.offset );               // Starts at the first uniform's offset
    EXPECT_EQ( binding.size, expected_size - binding.offset );// Covers the whole range used by compute

    // 4. Set Data
    block.set( h_time.set_value( 123.45f ) );
    block.set( h_frame.set_value( 99 ) );

    // 5. Verify Memory Contents
    const auto* raw_data = static_cast<const uint8_t*>( block.data() );
    EXPECT_FLOAT_EQ( *reinterpret_cast<const float*>( raw_data + h_time.offset ), 123.45f );
    EXPECT_EQ( *reinterpret_cast<const int*>( raw_data + h_frame.offset ), 99 );
}

// Test reflection of a struct uniform
TEST_F( PipelineManagerTestFixture, UniformBlockStruct ) {
    struct MyParams {
        float intensity;
        int mode;
    };

    pipeline_manager_->set_virtual_file( "struct_uniform.slang",
                                         "struct MyParams { float intensity; int mode; };\n" +
                                             make_compute_shader( "// body", "uniform MyParams params" ) );

    const aloe::ShaderCompileInfo shader{ .name = "struct_uniform.slang", .entry_point = "compute_main" };
    const auto result = compile_and_validate( { shader } );
    ASSERT_TRUE( result ) << result.error();

    auto h_params = pipeline_manager_->get_uniform_handle<MyParams>( *result, VK_SHADER_STAGE_COMPUTE_BIT, "params" );
    EXPECT_EQ( h_params.offset, 0 );

    auto& block = pipeline_manager_->get_uniform_block( *result );
    // Slang should ensure standard layout packing (e.g., std140/std430 rules might apply depending on context,
    // but for push constants it's often tightly packed or uses explicit offsets). Assuming tight packing here.
    ASSERT_EQ( block.size(), sizeof( MyParams ) );
    ASSERT_EQ( block.get_bindings().size(), 1 );

    const auto& binding = block.get_bindings().front();
    EXPECT_EQ( binding.stage_flags, VK_SHADER_STAGE_COMPUTE_BIT );
    EXPECT_EQ( binding.offset, 0 );
    EXPECT_EQ( binding.size, sizeof( MyParams ) );

    MyParams test_data = { 0.75f, 2 };
    block.set( h_params.set_value( test_data ) );

    const auto* raw_data = static_cast<const uint8_t*>( block.data() ) + h_params.offset;
    const auto* cpu_data = reinterpret_cast<const uint8_t*>( &test_data );
    const auto raw_slice = std::span( raw_data, binding.size );
    const auto cpu_slice = std::span( cpu_data, binding.size );
    EXPECT_TRUE( std::ranges::equal( raw_slice, cpu_slice ) );
}

// Test pipeline compilation fails if uniforms exceed maxPushConstantSize
TEST_F( PipelineManagerTestFixture, UniformBlockExceedsSizeLimit ) {
    const auto large_size = device_->get_physical_device_limits().maxPushConstantsSize + 4;

    // Define a struct that is guaranteed to be larger than the limit
    const auto shader_code =
        std::format( "struct BigStruct {{ float data[{}]; }};\n", large_size / sizeof( float ) + 1 ) +
        make_compute_shader( "// body", "uniform BigStruct large_uniform" );

    pipeline_manager_->set_virtual_file( "large_uniform.slang", shader_code );

    const aloe::ShaderCompileInfo shader{ .name = "large_uniform.slang", .entry_point = "compute_main" };
    const auto result = compile_and_validate( { shader } );

    ASSERT_FALSE( result );// Expect failure
    EXPECT_NE( result.error().find( "exceeds device limit" ), std::string::npos );
    EXPECT_NE( result.error().find( std::to_string( device_->get_physical_device_limits().maxPushConstantsSize ) ),
               std::string::npos );
}


// Test case for a shader with no uniforms at all
TEST_F( PipelineManagerTestFixture, UniformBlockNoUniforms ) {
    pipeline_manager_->set_virtual_file( "no_uniform.slang", make_compute_shader( "// body" ) );

    const aloe::ShaderCompileInfo shader{ .name = "no_uniform.slang", .entry_point = "compute_main" };
    const auto result = compile_and_validate( { shader } );
    ASSERT_TRUE( result ) << result.error();

    // Get the block - should exist but be empty
    auto& block = pipeline_manager_->get_uniform_block( *result );
    EXPECT_EQ( block.size(), 0 );               // Expect zero size
    EXPECT_TRUE( block.get_bindings().empty() );// Expect no bindings
}

// Compile a simple shader from file and verify success and valid output
TEST_F( PipelineManagerTestFixture, SimpleBufferBindingWorks ) {
    // Compile the test shader, ensure it is valid.
    const auto shader = aloe::ShaderCompileInfo{ .name = "test.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    // Allocate a buffer
    const auto buffer = resource_manager_->create_buffer( {
        .size = sizeof(float) * 128,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .name = "Test Buffer"
    } );

    const auto result = pipeline_manager_->bind_buffer( *resource_manager_, buffer );
    EXPECT_EQ(result, VK_SUCCESS);
}

TEST_F( PipelineManagerTestFixture, InvalidBufferBindingFailsGracefully ) {
    // Compile the test shader, ensure it is valid.
    const auto shader = aloe::ShaderCompileInfo{ .name = "test.slang", .entry_point = "main" };
    const auto handle = compile_and_validate( { shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    // Allocate a fake buffer, which does not exist.
    const auto buffer = aloe::BufferHandle(1, 5, 23);
    const auto result = pipeline_manager_->bind_buffer( *resource_manager_, buffer );
    EXPECT_NE(result, VK_SUCCESS);
}

// todo:
// 1. Test usage of a shared resource (i.e. aloe::BufferHandle) as a uniform :y:
// 2. Test usage of overlapping ranges once graphics pipelines are supported :y:
// 3. Test usage of equivalent ranges once graphics pipelines are supported :n: