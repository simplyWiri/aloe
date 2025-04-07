#pragma once

#include <tl/expected.hpp>

#include <memory>

#define VK_ENABLE_BETA_EXTENSIONS
#include <volk.h>

typedef struct VmaAllocator_T* VmaAllocator;

namespace aloe {

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

class Device {
public:
    struct DebugInformation {
        uint32_t num_verbose{ 0 };
        uint32_t num_info{ 0 };
        uint32_t num_warning{ 0 };
        uint32_t num_error{ 0 };
        uint32_t num_unknown{ 0 };
    };

    struct Queue {
        VkQueue queue;
        VkQueueFamilyProperties properties;
        uint32_t family_index;
    };
private:
    struct PhysicalDevice {
        VkPhysicalDevice physical_device;

        VkPhysicalDeviceProperties props;
        VkPhysicalDeviceFeatures features;
        VkPhysicalDeviceMemoryProperties mem_properties;

        std::vector<VkQueueFamilyProperties> queue_families;

        bool viable_device = true;
    };

    static DebugInformation debug_info_;

    bool enable_validation_ = false;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    std::vector<PhysicalDevice> physical_devices_;
    VkDevice device_ = VK_NULL_HANDLE;
    std::vector<Queue> queues_;
    VmaAllocator allocator_ = VK_NULL_HANDLE;

public:
    // Attempt to construct a new device
    static tl::expected<std::unique_ptr<Device>, VkResult> create_device( DeviceSettings settings );
    ~Device();


    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physical_device() const { return physical_devices_.front().physical_device; }
    VkDevice device() const { return device_; }
    VmaAllocator allocator() const { return allocator_; }
    bool validation_enabled() const { return enable_validation_; }
    std::vector<Queue> queues_by_capability( VkQueueFlagBits capability ) const;

    static const DebugInformation& debug_info() { return debug_info_; }

private:
    static VkResult create_instance( Device& device, const DeviceSettings& settings );
    static VkResult pick_physical_device( Device& device, const DeviceSettings& settings );
    static VkResult create_logical_device( Device& device, const DeviceSettings& settings );
    static void gather_queues( Device& device );
    static VkResult create_allocator( Device& device );
    static VkBool32 debug_callback( VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                                    VkDebugUtilsMessageTypeFlagsEXT message_type,
                                    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
                                    void* );


    Device() = default;
};

}// namespace aloe