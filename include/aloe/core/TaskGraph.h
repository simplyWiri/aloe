#pragma once

#include <aloe/core/CommandList.h>
#include <aloe/core/Device.h>
#include <aloe/core/Handles.h>

#include <functional>
#include <string>
#include <vector>

namespace aloe {

struct TaskDesc {
    std::string name;
    VkQueueFlagBits queue_type = VK_QUEUE_GRAPHICS_BIT;
    std::vector<ResourceUsage> resources;
    std::function<void( CommandList& )> execute_fn;
};

class TaskGraph {
    friend class Device;

    struct Task {
        // todo: sync primitives
        std::function<void( CommandList& )> execute_fn;
    };

private:
    Device& device_;
    PipelineManager& pipeline_manager_;
    ResourceManager& resource_manager_;
    std::vector<TaskDesc> task_descs_;
    std::vector<Task> tasks_;

    SimulationState state_;

    Device::Queue queue_ = {};
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
public:
    ~TaskGraph();

    TaskGraph( TaskGraph& ) = delete;
    TaskGraph& operator=( const TaskGraph& other ) = delete;
    TaskGraph( TaskGraph&& ) = delete;
    TaskGraph& operator=( TaskGraph&& other ) = delete;

    void add_task( TaskDesc&& task );
    void clear();  // Removes all tasks from the graph
    void compile();// Resolves dependencies, resource transitions, and synchronization
    void execute();// Executes all tasks in order

protected:
    explicit TaskGraph( Device& device, PipelineManager& pipeline_manager, ResourceManager& resource_manager );

private:
    void validate_task(CommandList& cmd, const TaskDesc& task_desc) const;
};

}// namespace aloe
