#pragma once

#include <aloe/core/handles.h>

#include <volk.h>

#include <memory>
#include <optional>

struct GLFWwindow;

namespace aloe {

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

struct Swapchain {
    Swapchain( const Device& device, SwapchainSettings settings );
    ~Swapchain();

    // Returns true we should exit the program
    bool poll_events();
    std::optional<RenderTarget> acquire_next_image( VkSemaphore image_available_semaphore );
    VkResult present( VkQueue queue, VkSemaphore wait_semaphore );

    VkExtent2D get_extent() const;
};

}// namespace aloe