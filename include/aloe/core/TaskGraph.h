#pragma once

#include <volk/volk.h>

#include <string>
#include <vector>
#include <functional>

namespace aloe {

struct TaskDesc {
    enum Queue { Graphics, Compute, Transfer };

    std::string name;
    Queue queue_type = Graphics;
    std::vector<ResourceUsageDesc> resources;
    // todo: `VkCommandBuffer` should be a `aloe::CommandList` once the interface exists.
    std::function<void(VkCommandBuffer)> execute_fn;
};

struct TaskGraph {
    struct Task {
        // semaphores / barriers which should be waited on
        std::function<void(VkCommandBuffer)> execute_fn;
    };


    void add_task(TaskDesc&& task);

    // Resolves all `TaskDesc`'s into concrete passes which can be executed, implicitly creates any resources
    // that are required for each task, and automates synchronisation injection between passes, to ensure
    // valid reads/writes for each resource.
    void compile();

    // Executes the task graph, runs through all of the tasks linearly, executing them one by one.
    void execute();

private:
    // todo: a way to get the current active command buffer for a given queue
    // todo: a way to manage command pools for each queue required by a particular graph
    // todo: a way for the frame graph to hook into this mechanism


    // Descriptions of a task which will be executed
    std::vector<TaskDesc> task_descs_;
    std::vector<Task> tasks_;
};

}
