#pragma once

#include <tl/expected.hpp>
#include <volk.h>

#include <memory>

struct GLFWwindow;

namespace aloe {

class Device;

struct SwapchainSettings {
    // Window:
    const char* title = "Aloe Window";
    int width = 1920;
    int height = 1080;

    // Use a HDR format
    bool use_hdr_surface = true;
};

struct RenderTarget {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

// An abstraction which managers the window, input, surface & swapchain for Vulkan
class Swapchain {
    constexpr static VkSurfaceFormatKHR hdr_target = { VK_FORMAT_A2B10G10R10_UNORM_PACK32,
                                                       VK_COLOR_SPACE_HDR10_ST2084_EXT };
    constexpr static VkSurfaceFormatKHR sdr_target = { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };

    const Device& device_;
    bool use_hdr_{ false };
    GLFWwindow* window_{ nullptr };
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    bool error_state_ = false;

    // Surface capabilities
    VkSurfaceCapabilitiesKHR capabilities_{};
    std::vector<VkSurfaceFormatKHR> formats_;
    std::vector<VkPresentModeKHR> present_modes_;
    bool hdr_supported_ = false;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> images_;
    std::vector<VkImageView> image_views_;

    uint32_t current_image_index_ = 0;

public:
    Swapchain( const Device& device, SwapchainSettings settings );
    ~Swapchain();

    Swapchain( Swapchain& ) = delete;
    Swapchain& operator=( const Swapchain& other ) = delete;

    Swapchain( Swapchain&& ) = delete;
    Swapchain& operator=( Swapchain&& other ) = delete;

    // Returns true we should exit the program
    bool poll_events();
    std::optional<RenderTarget> acquire_next_image(VkSemaphore image_available_semaphore);
    VkResult present( VkQueue queue, VkSemaphore wait_semaphore );

    GLFWwindow* window() const { return window_; }

protected:
    void resize();
    VkResult load_surface_capabilities();
    VkResult build_swapchain();

private:
    // Internal helpers for construction
    static VkResult create_window( Swapchain& swapchain, const SwapchainSettings& settings );
    static VkResult create_surface( const Device& device, Swapchain& swapchain );
    static void setup_callbacks( Swapchain& swapchain );

    // GLFW Callbacks
    static void glfw_error_callback( int error_code, const char* message );
    static void window_resized( GLFWwindow* window, int width, int height );
};

}// namespace aloe