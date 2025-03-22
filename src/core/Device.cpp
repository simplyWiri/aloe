#include <aloe/core/Device.h>
#include <aloe/util/log.h>

namespace aloe {

tl::expected<std::unique_ptr<Device>, VkResult> Device::create_device( DeviceSettings settings ) {
    // std::make_unique doesn't work when the class has a private constructor.
    auto device = std::unique_ptr<Device>( new Device() );

    // Initialize volk, create our VkInstance
    auto result = create_instance( *device, settings );
    if ( result != VK_SUCCESS ) { return tl::make_unexpected( result ); }

    return device;
}


VkResult Device::create_instance( Device& device, const DeviceSettings& settings ) {
    auto result = volkInitialize();
    if ( result != VK_SUCCESS ) return result;

    VkApplicationInfo app_info{ .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                .pApplicationName = settings.name,
                                .applicationVersion = settings.version,
                                .pEngineName = "aloe",
                                .engineVersion = VK_MAKE_VERSION( 1, 0, 0 ),
                                .apiVersion = VK_API_VERSION_1_2 };

    constexpr std::array validation_layers = { "VK_LAYER_KHRONOS_validation" };
    constexpr std::array instance_extensions = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
        VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME,
    };

    VkInstanceCreateInfo instance_info{ .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                        .flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
                                        .pApplicationInfo = &app_info,
                                        .enabledExtensionCount = static_cast<uint32_t>( instance_extensions.size() ),
                                        .ppEnabledExtensionNames = instance_extensions.data() };

    // Debug Utilities and Layers
    VkDebugUtilsMessengerCreateInfoEXT debug_info{};
    if ( settings.enable_validation ) {
        debug_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debug_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debug_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debug_info.pUserData = &device;
        debug_info.pfnUserCallback = Device::debug_callback;


        instance_info.enabledLayerCount = static_cast<uint32_t>( validation_layers.size() );
        instance_info.ppEnabledLayerNames = validation_layers.data();
        instance_info.pNext = &debug_info;
    }

    result = vkCreateInstance( &instance_info, nullptr, &device.instance_ );
    if ( result == VK_SUCCESS ) { volkLoadInstance( device.instance_ ); }

    log_write( LogLevel::Trace,
               "Successfully loaded Volk & created Vulkan instance, validation is {:s}, using instance layers: {}, "
               "and api version {}.{}.{}",
               settings.enable_validation ? "enabled" : "disabled",
               instance_extensions,
               VK_API_VERSION_MAJOR( app_info.apiVersion ),
               VK_API_VERSION_MINOR( app_info.apiVersion ),
               VK_API_VERSION_PATCH( app_info.apiVersion ) );

    return result;
}

VkBool32 Device::debug_callback( VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                                 VkDebugUtilsMessageTypeFlagsEXT,
                                 const VkDebugUtilsMessengerCallbackDataEXT*,
                                 void* user_info ) {
    auto& debug_info = static_cast<Device*>( user_info )->debug_info_;

    switch ( message_severity ) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT: ++debug_info.num_verbose_; break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT: ++debug_info.num_error_; break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT: ++debug_info.num_warning_; break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT: ++debug_info.num_info_; break;
        default: ++debug_info.num_unknown_; break;
    }

    return VK_FALSE;
}


Device::~Device() {
    if ( debug_messenger_ != VK_NULL_HANDLE ) {
        vkDestroyDebugUtilsMessengerEXT( instance_, debug_messenger_, nullptr );
    }

    if ( instance_ != VK_NULL_HANDLE ) { vkDestroyInstance( instance_, nullptr ); }
}


}// namespace aloe