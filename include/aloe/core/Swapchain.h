#pragma once

#include <memory>

#include <tl/expected.hpp>
#include <volk.h>

struct GLFWwindow;

namespace aloe {

class Device;

// An abstraction which managers the window, input, surface & swapchain for Vulkan
class Swapchain {
    GLFWwindow* window_{ nullptr };

public:
    static tl::expected<std::unique_ptr<Swapchain>, VkResult> create_swapchain( const Device& device );
    ~Swapchain();


private:
    static VkResult create_window( Swapchain& swapchain );
    static void glfw_error_callback( int error_code, const char* message );

    Swapchain() = default;
};

}// namespace aloe