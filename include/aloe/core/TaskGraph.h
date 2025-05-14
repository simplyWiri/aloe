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
    std::function<void(CommandList&)> execute_fn;
};

struct TaskGraph {
    void add_task(TaskDesc&& task);

    // Resolves all `TaskDesc`'s into concrete passes which can be executed, implicitly creates any resources
    // that are required for each task, and automates synchronisation injection between passes, to ensure
    // valid reads/writes for each resource.
    void compile();

    // Executes the task graph, runs through all of the tasks linearly, executing them one by one.
    void execute();
};

}
