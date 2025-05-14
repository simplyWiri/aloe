#pragma once

#include <aloe/core/TaskGraph.h>

#include <volk/volk.h>

#include <functional>
#include <string>
#include <vector>

namespace aloe {

struct FrameGraph : TaskGraph {
    // Wraps `TaskGraph::execute()`, acquires an image prior - and blits the resulting output from the task graph
    // to the image, and presents it.
    void execute();

    void set_output_image( ImageHandle handle );
};

}// namespace aloe
