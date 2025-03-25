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
        pipeline_manager_ = std::shared_ptr<aloe::PipelineManager>( new aloe::PipelineManager( {
            .device = *device_,
            .root_paths = { "resources" },
            .debug_info = true,
        } ) );

        spirv_tools_.SetMessageConsumer(
            [&]( spv_message_level_t level, const char* source, const spv_position_t& position, const char* message ) {
                EXPECT_TRUE( level > SPV_MSG_WARNING )
                    << source << " errored at line: " << position.line << ", with error: " << message;
            } );
    }
};

TEST_F( PipelineManagerTestFixture, BasicShaderCompilation ) {
    const auto result = pipeline_manager_->compile_shader( {
        .name = "test.slang",
        .entry_point = "main",
    } );

    ASSERT_TRUE( result.has_value() ) << "Actual result: " << result.error();
    EXPECT_TRUE( spirv_tools_.Validate( *result ) );
}

TEST_F( PipelineManagerTestFixture, BasicShaderInline ) {
    const auto result = pipeline_manager_->compile_shader( {
        .name = "example_inline.slang",
        .shader_code = R"(
[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 global_id : SV_DispatchThreadID, uint3 local_id : SV_GroupThreadID) { }
        )",
        .entry_point = "main",
    } );

    ASSERT_TRUE( result.has_value() ) << "Actual result: " << result.error();
    EXPECT_TRUE( spirv_tools_.Validate( *result ) );
}

TEST_F( PipelineManagerTestFixture, ShaderModules ) {
    const auto result = pipeline_manager_->compile_shader( {
        .name = "example2_inline.slang",
        .shader_code = R"(
import test;

[shader("vertex")]
void main() {
    int result = add_test(1, 5);
})",
        .entry_point = "main",
    } );

    ASSERT_TRUE( result.has_value() ) << "Actual result: " << result.error();
    EXPECT_TRUE( spirv_tools_.Validate( *result ) );
}

TEST_F( PipelineManagerTestFixture, ShaderDefines ) {
    const auto result = pipeline_manager_->compile_shader( {
        .name = "example3_inline.slang",
        .shader_code = R"(
[shader("vertex")]
void main() {
    int result = TEST_DEFINE;
})",
        .entry_point = "main",
        .defines = {
            { "TEST_DEFINE", "1" },
        },
    } );

    ASSERT_TRUE( result.has_value() ) << "Actual result: " << result.error();
    EXPECT_TRUE( spirv_tools_.Validate( *result ) );
}
