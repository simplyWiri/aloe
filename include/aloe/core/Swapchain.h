#pragma once

#include <memory>

#include <tl/expected.hpp>
#include <volk.h>

struct GLFWwindow;

namespace aloe {

class Device;

struct SwapchainSettings {
    // Window:
    const char* title = "Aloe Window";
    int width = 1920;
    int height = 1080;
};

// An abstraction which managers the window, input, surface & swapchain for Vulkan
class Swapchain {
    constexpr static VkSurfaceFormatKHR hdr_target = { VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_ST2084_EXT };
    constexpr static VkSurfaceFormatKHR sdr_target = { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };

    const Device& device_;
    GLFWwindow* window_{ nullptr };
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;

    // Surface capabilities
    VkSurfaceCapabilitiesKHR capabilities_;
    std::vector<VkSurfaceFormatKHR> formats_;
    std::vector<VkPresentModeKHR> present_modes_;

public:
    static tl::expected<std::unique_ptr<Swapchain>, VkResult> create_swapchain( const Device& device,
                                                                                SwapchainSettings settings );
    ~Swapchain();

protected:
    void resize( int new_width, int new_height );
    VkResult load_surface_capabilities();

private:
    // Internal helpers for construction
    static VkResult create_window( Swapchain& swapchain, const SwapchainSettings& settings );
    static VkResult create_surface( const Device& device, Swapchain& swapchain );
    static void setup_callbacks( Swapchain& swapchain );

    // GLFW Callbacks
    static void glfw_error_callback( int error_code, const char* message );
    static void window_position_callback( GLFWwindow* window, int x, int y );
    static void window_resized( GLFWwindow* window, int width, int height );

    Swapchain( const Device& device );
};

}// namespace aloe