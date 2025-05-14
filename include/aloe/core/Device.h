#pragma once

#include <aloe/core/handles.h>

#include <memory>
#include <vector>

#define VK_ENABLE_BETA_EXTENSIONS
#include <volk.h>
#include <vma/vma.h>

namespace aloe {

struct SwapchainSettings;

struct DeviceSettings {
    const char* name = "aloe application";
    uint32_t version = VK_MAKE_VERSION( 1, 0, 0 );

    bool enable_validation = true;
    bool headless = false;

    std::vector<const char*> device_extensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,          VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,  VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_MAINTENANCE1_EXTENSION_NAME,       VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,
        VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME,
    };
};

struct Device {
    struct Queue {
        VkQueue queue;
        VkQueueFamilyProperties properties;
        uint32_t family_index;
    };

    Device( DeviceSettings settings );

    std::vector<Queue> find_queues( VkQueueFlagBits capability ) const;

    std::shared_ptr<PipelineManager> make_pipeline_manager(const std::vector<std::string>& root_paths);
    std::shared_ptr<ResourceManager> make_resource_manager();
    std::shared_ptr<Swapchain> make_swapchain(const SwapchainSettings& settings);
    TaskGraph make_task_graph();
    FrameGraph make_frame_graph();
};

}// namespace aloe