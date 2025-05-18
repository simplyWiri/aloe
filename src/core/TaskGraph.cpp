#include <aloe/core/TaskGraph.h>
#include <aloe/util/log.h>

#include <set>
#include <unordered_set>

namespace aloe {

TaskGraph::TaskGraph( Device& device, PipelineManager& pipeline_manager, ResourceManager& resource_manager )
    : device_( device )
    , pipeline_manager_( pipeline_manager )
    , resource_manager_( resource_manager ) {
}

void TaskGraph::validate_task( CommandList& cmd, const TaskDesc& task_desc ) const {
    // Check that the set of bound resources, is a subset of the resources which we declared in the task description.
    {
        std::set<ResourceUsage> all_bound_resources;
        for ( const auto& handle : cmd.bound_pipelines() ) {
            const auto& resources = pipeline_manager_.get_bound_resources( handle );
            all_bound_resources.insert( resources.begin(), resources.end() );
        }

        for ( const auto& expected : task_desc.resources ) {
            if ( all_bound_resources.contains( expected ) ) continue;

            log_write( LogLevel::Warn,
                       "resource expected by task '%s' was not bound by any pipeline.",
                       task_desc.name );
        }
    }
}

TaskGraph::~TaskGraph() {
    if ( command_pool_ != VK_NULL_HANDLE ) { vkDestroyCommandPool( device_.device(), command_pool_, nullptr ); }
}

void TaskGraph::add_task( TaskDesc&& task ) {
    task_descs_.emplace_back( std::move( task ) );
}

void TaskGraph::clear() {
    task_descs_.clear();
    tasks_.clear();

    if ( command_pool_ != VK_NULL_HANDLE ) {
        vkDestroyCommandPool( device_.device(), command_pool_, nullptr );
        command_pool_ = VK_NULL_HANDLE;
    }
}

void TaskGraph::compile() {
    int queue_flags = 0;

    for ( const auto& task_desc : task_descs_ ) {
        // Verify that the `bound_resource.resource`'s are unique, the same resource can not be referred to twice in the
        // same task.
        {
            std::unordered_set<std::variant<BufferHandle, ImageHandle>> seen_resources;
            for ( const auto& bound_resource : task_desc.resources ) {
                const auto inserted_resource = seen_resources.insert( bound_resource.resource ).second;
                if ( inserted_resource ) continue;

                log_write( LogLevel::Error, "resource used more than once in task %s'", task_desc.name );
                return;
            }
        }

        // Create bindings for each resource
        for ( const auto& bound_resource : task_desc.resources ) {
            const auto slot = resource_manager_.bind_resource( bound_resource );
            if ( slot == std::nullopt ) {
                log_write( LogLevel::Error, "failed to allocate a slot for resource" );
                return;
            }
        }

        Task task;
        task.execute_fn = task_desc.execute_fn;
        tasks_.emplace_back( std::move( task ) );

        queue_flags |= static_cast<int>( task_desc.queue_type );
    }

    pipeline_manager_.bind_slots();

    // Create a command pool for the required queue family
    queue_ = device_.find_queues( static_cast<VkQueueFlagBits>( queue_flags ) ).front();

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_.family_index;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    vkCreateCommandPool( device_.device(), &pool_info, nullptr, &command_pool_ );

    VkCommandBufferAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool_,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    vkAllocateCommandBuffers( device_.device(), &alloc_info, &command_buffer_ );
}

void TaskGraph::execute() {
    using namespace std::chrono_literals;

    // Update `SimulationState`:
    const auto current_time = std::chrono::high_resolution_clock::now().time_since_epoch();
    const auto micros_since_epoch = std::chrono::duration_cast<std::chrono::microseconds>( current_time );

    state_.sim_index++;
    state_.delta_time = state_.time_since_epoch != 0us ? micros_since_epoch - state_.time_since_epoch : 0us;
    state_.time_since_epoch = micros_since_epoch;

    VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    vkBeginCommandBuffer( command_buffer_, &begin_info );

    for ( std::size_t i = 0; i < tasks_.size(); ++i ) {
        const auto& desc = task_descs_[i];
        auto& task = tasks_[i];
        {
            CommandList task_list{ pipeline_manager_, resource_manager_, desc.name.c_str(), command_buffer_, state_ };
            task.execute_fn( task_list );

            validate_task( task_list, desc );
        }
    }

    // Submit the command buffer
    {
        vkEndCommandBuffer( command_buffer_ );

        const VkSubmitInfo submit_info{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffer_,
        };

        vkQueueSubmit( queue_.queue, 1, &submit_info, VK_NULL_HANDLE );
        vkQueueWaitIdle( queue_.queue );
    }
}

}// namespace aloe