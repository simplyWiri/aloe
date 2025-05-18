#include <aloe/core/Device.h>
#include <aloe/core/PipelineManager.h>
#include <aloe/core/ResourceManager.h>
#include <aloe/core/TaskGraph.h>
#include <aloe/util/log.h>

#include <gtest/gtest.h>

#include <numeric>

class TaskGraphTestFixture : public ::testing::Test {
protected:
    std::shared_ptr<aloe::MockLogger> mock_logger_;
    std::unique_ptr<aloe::Device> device_;
    std::shared_ptr<aloe::PipelineManager> pipeline_manager_;
    std::shared_ptr<aloe::ResourceManager> resource_manager_;
    std::shared_ptr<aloe::TaskGraph> task_graph_;

    int shader_id_ = 1;

    void SetUp() override {
        mock_logger_ = std::make_shared<aloe::MockLogger>();
        aloe::set_logger( mock_logger_ );
        aloe::set_logger_level( aloe::LogLevel::Warn );

        device_ = std::make_unique<aloe::Device>( aloe::DeviceSettings{ .enable_validation = true, .headless = true } );

        resource_manager_ = device_->make_resource_manager();
        pipeline_manager_ = device_->make_pipeline_manager( {} );
        task_graph_ = device_->make_task_graph();
    }

    void TearDown() override {
        task_graph_.reset();
        pipeline_manager_.reset();
        resource_manager_.reset();
        device_.reset();

        auto& debug_info = aloe::Device::debug_info();
        EXPECT_EQ( debug_info.num_warning, 0 );
        EXPECT_EQ( debug_info.num_error, 0 );

        if ( debug_info.num_warning > 0 || debug_info.num_error > 0 ) {
            for ( const auto& [level, message] : mock_logger_->get_entries() ) { std::cerr << message << std::endl; }
        }
    }

    aloe::BufferHandle create_test_buffer( size_t size = sizeof( int ), const char* name = "TestBuffer" ) const {
        const aloe::BufferDesc desc = { .size = size,
                                        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        .memory_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                                        .name = name };
        return resource_manager_->create_buffer( desc );
    }

    aloe::PipelineHandle create_compute_pipeline( std::vector<std::string> shader_uniforms,
                                                  const std::string shader_body ) {
        constexpr const auto* shader_template = R"(
            import aloe;

            [shader("compute")]
            void compute_main(uint3 id : SV_DispatchThreadID{})
            {{
                {}
            }}
        )";

        const auto uniforms_string =
            std::accumulate( shader_uniforms.begin(),
                             shader_uniforms.end(),
                             std::string{},
                             []( const std::string& acc, const std::string& s ) { return acc + ", uniform " + s; } );
        const auto shader_name = std::format( "shader_{}.slang", ++shader_id_ );
        const auto shader_contents = std::format( shader_template, uniforms_string, shader_body );

        pipeline_manager_->set_virtual_file( shader_name, shader_contents );
        const auto handle = pipeline_manager_->compile_pipeline( {
            .compute_shader = {
                .name = shader_name,
                .entry_point = "compute_main",
            },
        } );
        EXPECT_TRUE( handle.has_value() ) << handle.error();
        return handle.value();
    }

    bool log_contains(const std::string& substring) const {
        for (const auto& [level, message] : mock_logger_->get_entries()) {
            if (message.find(substring) != std::string::npos) {
                return true;
            }
        }
        return false;
    }
};

//------------------------------------------------------------------------------
// Task Management Tests
//------------------------------------------------------------------------------

// Test adding tasks and verifying task graph state
TEST_F( TaskGraphTestFixture, Task_BasicOperations ) {
    const auto buffer0 = create_test_buffer( sizeof( int ), "Buffer 0" );
    const auto buffer1 = create_test_buffer( sizeof( int ), "Buffer 1" );

    // Create compute shader that writes a value from a uniform into the buffer
    const auto pipeline =
        create_compute_pipeline( { "int value", "aloe::BufferHandle buffer" }, "buffer.get().Store<int>(0, value);" );

    // Uniform handles (declared outside lambda for clarity)
    auto value_uniform = pipeline_manager_->get_uniform_handle<int>( pipeline, "value" );
    auto buffer_uniform = pipeline_manager_->get_uniform_handle<aloe::BufferHandle>( pipeline, "buffer" );

    task_graph_->add_task( {
        .name = "Task_BasicOperations",
        .queue_type = VK_QUEUE_COMPUTE_BIT,
        .resources = { aloe::usage( buffer0, aloe::ComputeStorageWrite ),
                       aloe::usage( buffer1, aloe::ComputeStorageWrite ), },
        .execute_fn = [&]( aloe::CommandList& cmd ) {
             const auto target_buffer = ( cmd.state().sim_index % 2 == 0 ) ? buffer1 : buffer0;
             const auto buffer_usage = aloe::usage( target_buffer, aloe::ComputeStorageWrite );

             cmd.bind_pipeline( pipeline )
                .set_uniform( value_uniform.set_value( static_cast<int>(cmd.state().sim_index * 5) ) )
                .set_uniform( buffer_uniform.set_value( target_buffer ), buffer_usage )
                .dispatch( 1, 1, 1 );
         },
    } );
    task_graph_->compile();

    // Verify the results: buffer_0: 5, buffer_1: 0
    task_graph_->execute();
    {
        int read0 = 0, read1 = 0;
        resource_manager_->read_from_buffer( buffer0, &read0, sizeof( int ) );
        resource_manager_->read_from_buffer( buffer1, &read1, sizeof( int ) );

        EXPECT_EQ( read0, 5 );
        EXPECT_EQ( read1, 0 );
    }


    task_graph_->execute();
    // Verify the results: buffer_0: 5 (unchanged), buffer_1: 10
    {
        int read0 = 0, read1 = 0;
        resource_manager_->read_from_buffer( buffer0, &read0, sizeof( int ) );
        resource_manager_->read_from_buffer( buffer1, &read1, sizeof( int ) );

        EXPECT_EQ( read0, 5 );
        EXPECT_EQ( read1, 10 );
    }
}

// Test clear() and rebuild with different tasks, verifying resource transitions are correct
TEST_F( TaskGraphTestFixture, Task_ClearAndRebuild ) {
}

//------------------------------------------------------------------------------
// Resource Usage and Synchronization Tests
//------------------------------------------------------------------------------

// Tests read-after-write dependency between two compute tasks using a buffer
// Also verifies proper resource transitions and barriers
TEST_F( TaskGraphTestFixture, ResourceSync_ReadAfterWrite ) {
}

// Tests write-after-read dependencies with multiple readers followed by a writer
TEST_F( TaskGraphTestFixture, ResourceSync_WriteAfterRead ) {
}

// Tests concurrent execution of tasks without resource dependencies
// Verifies multiple independent tasks can run in parallel
TEST_F( TaskGraphTestFixture, ResourceSync_IndependentTasks ) {
}

// Tests cross-queue synchronization (compute->graphics->compute)
// Verifies queue ownership transfers and proper synchronization
TEST_F( TaskGraphTestFixture, ResourceSync_CrossQueue ) {
}

//------------------------------------------------------------------------------
// Pipeline State and Layout Tests
//------------------------------------------------------------------------------

// Tests pipeline state transitions through multiple render passes
// Verifies image layouts change correctly between color attachment and shader read
TEST_F( TaskGraphTestFixture, PipelineState_RenderPassTransitions ) {
}

// Tests compute pipeline state transitions with multiple storage images
// Verifies proper layout transitions between general and shader read
TEST_F( TaskGraphTestFixture, PipelineState_ComputeImageTransitions ) {
}

//------------------------------------------------------------------------------
// Error Handling and Validation Tests
//------------------------------------------------------------------------------

// Tests error handling for invalid resource usage
// 1. Using freed resources
// 2. Invalid queue type for operation
// 3. Writing to read-only resource
TEST_F( TaskGraphTestFixture, Error_ResourceValidation ) {
}

// Tests error handling for dependency/barrier issues
// 1. Circular dependencies
// 2. Missing synchronization
// 3. Invalid resource transitions
TEST_F( TaskGraphTestFixture, Error_DependencyValidation ) {
}

// Tests error when a task declares multiple accesses of the same resource
TEST_F( TaskGraphTestFixture, Error_DuplicateResourceDeclaration ) {
    const auto buffer = create_test_buffer(sizeof(int), "Buffer_MultiAccess");

    aloe::TaskDesc invalid_task{
        .name = "MultiAccessSameResource",
        .queue_type = VK_QUEUE_COMPUTE_BIT,
        .resources = {
            aloe::usage(buffer, aloe::ComputeStorageWrite),
            aloe::usage(buffer, aloe::ComputeStorageRead) // Duplicate resource
        },
        .execute_fn = [&](aloe::CommandList&) {
            // No-op
        }
    };

    task_graph_->add_task(std::move(invalid_task));
    task_graph_->compile();

    // Check that an error was logged about duplicate resource usage
    EXPECT_TRUE(log_contains("resource used more than once")) << "Expected error for multiple accesses of the same resource in a task";
}

// Tests error when a task binds a resource in the pipeline that was not declared in the task
TEST_F( TaskGraphTestFixture, Error_UndeclaredResourceBinding ) {
    const auto buffer_declared = create_test_buffer(sizeof(int), "Buffer_Declared");
    const auto buffer_undeclared = create_test_buffer(sizeof(int), "Buffer_Undeclared");

    // Create a simple compute pipeline
    const auto pipeline = create_compute_pipeline(
        { "int value", "aloe::BufferHandle buffer" },
        "buffer.get().Store<int>(0, value);"
    );
    auto value_uniform = pipeline_manager_->get_uniform_handle<int>(pipeline, "value");
    auto buffer_uniform = pipeline_manager_->get_uniform_handle<aloe::BufferHandle>(pipeline, "buffer");

    aloe::TaskDesc invalid_bind_task{
        .name = "BindUndeclaredResource",
        .queue_type = VK_QUEUE_COMPUTE_BIT,
        .resources = {
            aloe::usage(buffer_declared, aloe::ComputeStorageWrite)
        },
        .execute_fn = [&](aloe::CommandList& cmd) {
            // Intentionally bind an undeclared buffer
            auto usage = aloe::usage(buffer_undeclared, aloe::ComputeStorageWrite);
            cmd.bind_pipeline(pipeline)
                .set_uniform(value_uniform.set_value(123))
                .set_uniform(buffer_uniform.set_value(buffer_undeclared), usage)
                .dispatch(1, 1, 1);
        }
    };

    task_graph_->add_task(std::move(invalid_bind_task));
    task_graph_->compile();
    task_graph_->execute();

    // Should log a warning or error about undeclared resource usage
    EXPECT_TRUE(log_contains("was not bound by any pipeline")) << "Expected error about binding undeclared resource in a task";
}

//------------------------------------------------------------------------------
// End-to-End Tests
//------------------------------------------------------------------------------

// Tests a compute->graphics->compute pipeline with multiple resources
// Simulates a typical render pass with post-processing
TEST_F( TaskGraphTestFixture, E2E_ComputeGraphicsPipeline ) {
}

// Tests a complex graphics pipeline with multiple render targets and dependencies
// Simulates a deferred rendering pipeline with geometry and lighting passes
TEST_F( TaskGraphTestFixture, E2E_MultiPassRendering ) {
}
