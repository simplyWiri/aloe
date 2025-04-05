#include <aloe/core/PipelineManager.h>
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
        aloe::set_logger_level( aloe::LogLevel::Trace );

        device_ = aloe::Device::create_device( { .enable_validation = false } ).value();
        pipeline_manager_ =
            std::make_shared<aloe::PipelineManager>( *device_, std::vector<std::string>{ "resources" } );

        spirv_tools_.SetMessageConsumer(
            [&]( spv_message_level_t level, const char* source, const spv_position_t& position, const char* message ) {
                EXPECT_TRUE( level > SPV_MSG_WARNING )
                    << source << " errored at line: " << position.line << ", with error: " << message;
            } );
    }
};

#define COMPUTE_ENTRY " [shader(\"compute\")] "

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
    pipeline_manager_->set_virtual_file( "test.slang", "module test; int add(int a, int b) { return 5 + a + b; }" );
    pipeline_manager_->set_virtual_file( "main_shader.slang",
                                         "import test;" COMPUTE_ENTRY " void main() { int x = add(1, 2); }" );

    const aloe::ShaderCompileInfo shader{ .name = "main_shader.slang", .entry_point = "main" };
    const auto handle = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();
}

TEST_F( PipelineManagerTestFixture, VirtualFileDependencyTriggersRecompilation ) {
    // Set up the virtual include file & a shader that dependents this file
    pipeline_manager_->set_virtual_file( "test.slang", "module test; int add(int a, int b) { return 5 + a + b; }" );
    pipeline_manager_->set_virtual_file( "main_shader.slang",
                                         "import test;" COMPUTE_ENTRY "void main() { int x = add(1, 2); }" );

    const aloe::ShaderCompileInfo shader{ .name = "main_shader.slang", .entry_point = "main" };
    const auto handle = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    uint64_t baseline_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( baseline_version, 1 );

    // Modify the shared include, this should automatically result in the pipeline being recompiled immediately
    pipeline_manager_->set_virtual_file( "test.slang", "module test; int add(int a, int b) { return 8 + a + b; }" );

    const uint64_t next_version = pipeline_manager_->get_pipeline_version( *handle );

    EXPECT_NE( baseline_version, next_version );
    EXPECT_EQ( next_version, baseline_version + 1 );
}

// Changing an unrelated virtual file should not trigger recompilation
TEST_F( PipelineManagerTestFixture, UnrelatedVirtualFileDependencyDoesntTriggerUpdate ) {
    // Set up the virtual include file & a shader that dependents this file
    pipeline_manager_->set_virtual_file( "test.slang", "module test; int add(int a, int b) { return 5 + a + b; }" );
    pipeline_manager_->set_virtual_file( "main_shader.slang", COMPUTE_ENTRY "void main() { int x = t; }" );

    const aloe::ShaderCompileInfo shader{ .name = "main_shader.slang", .entry_point = "main" };
    const auto handle = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    uint64_t baseline_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( baseline_version, 1 );

    // Modify the shared include, this should not result in the pipeline being recompiled - because `main_shader.slang`
    // does not depend on `test` at all.
    pipeline_manager_->set_virtual_file( "test.slang", "module test; int add(int a, int b) { return 8 + a + b; }" );

    const uint64_t next_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( baseline_version, next_version );
}

// Updating a dependent file causes the pipeline to automatically recompile
TEST_F( PipelineManagerTestFixture, DependencyUpdateRecompilesFinalShader ) {
    // Set up shader module graph: `main` depends on `mid`
    pipeline_manager_->set_virtual_file( "mid.slang", "module mid; int square(int x) { return x * x; }" );
    pipeline_manager_->set_virtual_file( "main.slang",
                                         "import mid;" COMPUTE_ENTRY "void main() { int x = square(4); }" );

    const aloe::ShaderCompileInfo shader{ .name = "main.slang", .entry_point = "main" };
    const auto handle = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    const uint64_t baseline_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( baseline_version, 1 );

    pipeline_manager_->set_virtual_file( "mid.slang", "module mid; int square(int x) { return x * x * x; }" );

    const uint64_t new_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( new_version, baseline_version + 1 );
}

// Changing a transitive dependency should recompile all dependents
TEST_F( PipelineManagerTestFixture, TransitiveDependencyUpdateRecompilesFinalShader ) {
    // main -> mid -> common
    pipeline_manager_->set_virtual_file( "common.slang", "module common; int add(int a, int b) { return a + b; }" );
    pipeline_manager_->set_virtual_file( "mid.slang",
                                         "module mid; import common; int triple(int x) { return add(x, add(x, x)); }" );
    pipeline_manager_->set_virtual_file( "main.slang",
                                         "import mid;" COMPUTE_ENTRY "void main() { int x = triple(3); }" );

    const aloe::ShaderCompileInfo shader{ .name = "main.slang", .entry_point = "main" };
    const auto handle = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    const uint64_t initial_version = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( initial_version, 1 );

    // Modify the transitive dependency: common.slang
    pipeline_manager_->set_virtual_file( "common.slang", "module common; int add(int a, int b) { return 1 + a + b; }" );

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
    pipeline_manager_->set_virtual_file( "shared_dep.slang", "module shared_dep; int val() { return 42; }" );
    pipeline_manager_->set_virtual_file( "mid_left.slang",
                                         "import shared_dep; module mid_left; int left() { return val(); }" );
    pipeline_manager_->set_virtual_file( "mid_right.slang",
                                         "import shared_dep; module mid_right; int right() { return val(); }" );
    pipeline_manager_->set_virtual_file( "main.slang",
                                         "import mid_left;import mid_right;" COMPUTE_ENTRY
                                         "void main() { int x = left() + right(); }" );

    const aloe::ShaderCompileInfo shader{ .name = "main.slang", .entry_point = "main" };
    const auto handle = pipeline_manager_->compile_pipeline( { .compute_shader = shader } );
    ASSERT_TRUE( handle.has_value() ) << handle.error();

    const uint64_t version_before = pipeline_manager_->get_pipeline_version( *handle );
    EXPECT_EQ( version_before, 1 );

    // Modify shared leaf dependency
    pipeline_manager_->set_virtual_file( "shared_dep.slang",
                                         "module shared_dep;\n"
                                         "int val() { return 1337; }" );

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
    pipeline_manager_->set_virtual_file( "shared.slang", "module shared; int val() { return 1; }" );

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
    pipeline_manager_->set_virtual_file( "shared.slang", "module shared; int val() { return 999; }" );

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
