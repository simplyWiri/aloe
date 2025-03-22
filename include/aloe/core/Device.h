#pragma once

#include <memory>

#include <tl/expected.hpp>
#include <volk.h>

namespace aloe {

struct DeviceSettings {
    const char* name = "aloe application";
    uint32_t version = VK_MAKE_VERSION( 1, 0, 0 );

    bool enable_validation = true;
};

class Device {
public:
    struct DebugInformation {
        uint32_t num_verbose_{ 0 };
        uint32_t num_info_{ 0 };
        uint32_t num_warning_{ 0 };
        uint32_t num_error_{ 0 };
        uint32_t num_unknown_{ 0 };
    };

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    DebugInformation debug_info_;

public:
    // Attempt to construct a new device
    static tl::expected<std::unique_ptr<Device>, VkResult> create_device( DeviceSettings settings );
    ~Device();


    const DebugInformation& debug_info() const { return debug_info_; }

private:
    static VkResult create_instance( Device& device, const DeviceSettings& settings );
    static VkBool32 debug_callback( VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                                    VkDebugUtilsMessageTypeFlagsEXT message_type,
                                    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
                                    void* );


    Device() = default;
};

}// namespace aloe