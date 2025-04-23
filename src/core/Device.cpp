#include <aloe/core/Device.h>
#include <aloe/core/PipelineManager.h>
#include <aloe/core/ResourceManager.h>
#include <aloe/core/Swapchain.h>
#include <aloe/util/log.h>
#include <aloe/util/vulkan_util.h>

#include <GLFW/glfw3.h>
#include <vma/vma.h>

#include <cassert>
#include <numeric>
#include <ranges>

namespace aloe {

Device::DebugInformation Device::debug_info_ = {};

Device::Device( DeviceSettings settings ) : enable_validation_( settings.enable_validation ) {
    // Reset our debug info
    Device::debug_info_ = {};

    // We do not need the `VK_KHR_SWAPCHAIN_EXTENSION_NAME` extensions if we are running a headless instance.
    if ( settings.headless ) {
        settings.device_extensions.erase( settings.device_extensions.begin(), settings.device_extensions.begin() + 1 );
    }

    // Initialize volk, create our VkInstance
    auto result = create_instance( *this, settings );
    if ( result != VK_SUCCESS ) { throw std::runtime_error( "Failed to create device instance" ); }

    result = pick_physical_device( *this, settings );
    if ( result != VK_SUCCESS ) { throw std::runtime_error( "Failed to find a physical device" ); }

    result = create_logical_device( *this, settings );
    if ( result != VK_SUCCESS ) { throw std::runtime_error( "Failed to make a logical device" ); }

    gather_queues( *this );

    result = create_allocator( *this );
    if ( result != VK_SUCCESS ) { throw std::runtime_error( "Failed to create a VMA Allocator" ); }
}

std::vector<Device::Queue> Device::queues_by_capability( VkQueueFlagBits capability ) const {
    return queues_ | std::views::filter( [&]( auto& q ) { return q.properties.queueFlags & capability; } ) |
        std::ranges::to<std::vector>();
}

std::shared_ptr<PipelineManager> Device::make_pipeline_manager( const std::vector<std::string>& root_paths ) {
    assert(pipeline_manager_ == nullptr);
    pipeline_manager_ = std::shared_ptr<PipelineManager>(new PipelineManager(*this, root_paths ));
    return pipeline_manager_;
}

std::shared_ptr<ResourceManager> Device::make_resource_manager() {
    assert(resource_manager_ == nullptr);
    resource_manager_ = std::shared_ptr<ResourceManager>(new ResourceManager(*this ));
    return resource_manager_;
}

std::shared_ptr<Swapchain> Device::make_swapchain( const SwapchainSettings& settings ) {
    assert(swapchain_ == nullptr);
    swapchain_ = std::shared_ptr<Swapchain>(new Swapchain( *this, settings ));
    return swapchain_;
}

VkResult Device::create_instance( Device& device, const DeviceSettings& settings ) {
    auto result = volkInitialize();
    if ( result != VK_SUCCESS ) return result;

    VkApplicationInfo app_info{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = settings.name,
        .applicationVersion = settings.version,
        .pEngineName = "aloe",
        .engineVersion = VK_MAKE_VERSION( 1, 0, 0 ),
        .apiVersion = VK_API_VERSION_1_2,
    };

    constexpr std::array validation_layers = { "VK_LAYER_KHRONOS_validation" };
    std::vector instance_extensions = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };

    // If we are running without a screen, we do not need any swapchain extensions, nor GLFW
    if ( !settings.headless ) {
        // We (try) initialize GLFW to create our instance
        if ( glfwInit() == GLFW_FALSE ) return VkResult::VK_ERROR_INITIALIZATION_FAILED;


        uint32_t count = 0;
        auto** glfw_exts = glfwGetRequiredInstanceExtensions( &count );

        instance_extensions.insert( instance_extensions.end(), glfw_exts, glfw_exts + count );
        instance_extensions.emplace_back( VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME );
    }

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
    if ( result == VK_SUCCESS ) {
        volkLoadInstance( device.instance_ );

        if ( settings.enable_validation ) {
            result = vkCreateDebugUtilsMessengerEXT( device.instance_, &debug_info, nullptr, &device.debug_messenger_ );
        }

        log_write( LogLevel::Trace,
                   "Successfully loaded Volk & created Vulkan instance, validation is {:s}, using instance layers: {}, "
                   "and api version {}.{}.{}",
                   settings.enable_validation ? "enabled" : "disabled",
                   instance_extensions,
                   VK_API_VERSION_MAJOR( app_info.apiVersion ),
                   VK_API_VERSION_MINOR( app_info.apiVersion ),
                   VK_API_VERSION_PATCH( app_info.apiVersion ) );
    } else {
        log_write( LogLevel::Error, "Failed to create a vulkan instance, error returned: {:s}", result );
    }

    return result;
}

VkResult Device::pick_physical_device( Device& device, const DeviceSettings& settings ) {
    // Retrieve all available physical devices
    auto physical_devices = get_enumerated_value<VkPhysicalDevice>(
        [&]( uint32_t* count, VkPhysicalDevice* devices ) {
            vkEnumeratePhysicalDevices( device.instance_, count, devices );
        },
        "Failed to enumerate Vulkan physical devices." );

    // Check each physical device for compatibility
    for ( const auto physical_device : physical_devices ) {
        auto& wrapper = device.physical_devices_.emplace_back( physical_device );

        // Gather features and properties of the current physical device
        vkGetPhysicalDeviceProperties( physical_device, &wrapper.props );
        vkGetPhysicalDeviceFeatures( physical_device, &wrapper.features );
        vkGetPhysicalDeviceMemoryProperties( physical_device, &wrapper.mem_properties );
        wrapper.queue_families = get_enumerated_value<VkQueueFamilyProperties>(
            [&]( uint32_t* count, VkQueueFamilyProperties* props ) {
                vkGetPhysicalDeviceQueueFamilyProperties( physical_device, count, props );
            },
            "Failed to enumerate physical device queue families" );

        uint64_t total_memory = 0;
        for ( uint32_t j = 0; j < wrapper.mem_properties.memoryHeapCount; ++j ) {
            if ( wrapper.mem_properties.memoryHeaps[j].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ) {
                total_memory += wrapper.mem_properties.memoryHeaps[j].size;
            }
        }

        auto extensions = get_enumerated_value<VkExtensionProperties>(
                              [&]( uint32_t* count, VkExtensionProperties* props ) {
                                  vkEnumerateDeviceExtensionProperties( physical_device, nullptr, count, props );
                              },
                              "Failed to enumerate physical device extensions" ) |
            std::views::transform( []( auto& e ) { return e.extensionName; } );

        auto queue_families = wrapper.queue_families | std::views::transform( []( auto& q ) {
                                  std::vector<const char*> qt;
                                  if ( q.queueFlags & VK_QUEUE_GRAPHICS_BIT ) { qt.emplace_back( "Graphics" ); }
                                  if ( q.queueFlags & VK_QUEUE_COMPUTE_BIT ) { qt.emplace_back( "Compute" ); }
                                  if ( q.queueFlags & VK_QUEUE_TRANSFER_BIT ) { qt.emplace_back( "Transfer" ); }
                                  return std::format( "{:d} ({:n:})", q.queueCount, qt );
                              } );

        log_write( LogLevel::Info, "Physical device: '{}'", std::string{ wrapper.props.deviceName } );
        log_write( LogLevel::Info,
                   "- Device Type: {}",
                   wrapper.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "Discrete GPU"
                                                                                    : "Integrated GPU" );
        log_write( LogLevel::Info,
                   "- API Version: {}.{}.{}",
                   VK_VERSION_MAJOR( wrapper.props.apiVersion ),
                   VK_VERSION_MINOR( wrapper.props.apiVersion ),
                   VK_VERSION_PATCH( wrapper.props.apiVersion ) );
        log_write( LogLevel::Info,
                   "- Driver Version: {}.{}.{}",
                   VK_VERSION_MAJOR( wrapper.props.driverVersion ),
                   VK_VERSION_MINOR( wrapper.props.driverVersion ),
                   VK_VERSION_PATCH( wrapper.props.driverVersion ) );

        log_write( LogLevel::Info, "- Total Device Memory: {} MB", total_memory / ( 1024 * 1024 ) );
        log_write( LogLevel::Trace, "- Queue Families: {}", queue_families );
        log_write( LogLevel::Trace, "- Supported Extensions: {}", extensions );


        for ( const auto& extension : settings.device_extensions ) {
            if ( std::ranges::find( extensions, std::string{ extension } ) == extensions.end() ) {
                log_write( LogLevel::Error, "Failed to find required extension: {}", extension );
                wrapper.viable_device = false;
            }
        }
    }

    // todo: Eventually we can sort physical devices by capabilities here, but for now I only ever run single-gpu
    // setups, so this is irrelevant.

    return device.physical_devices_.front().viable_device ? VK_SUCCESS : VK_ERROR_INITIALIZATION_FAILED;
}

VkResult Device::create_logical_device( Device& device, const DeviceSettings& settings ) {
    const auto& physical_device = device.physical_devices_.front();

    const float priority = 1.0f;
    auto queue_infos = std::views::iota( 0 ) | std::views::take( physical_device.queue_families.size() ) |
        std::views::transform( [&]( auto index ) {
                           return VkDeviceQueueCreateInfo{
                               .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                               .queueFamilyIndex = static_cast<uint32_t>( index ),
                               .queueCount = 1,
                               .pQueuePriorities = &priority,
                           };
                       } ) |
        std::ranges::to<std::vector>();

    // We use dynamic rendering, sync2 + Core 1.2 features

    VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamic_rendering{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
        .dynamicRendering = VK_TRUE,
    };

    VkPhysicalDeviceSynchronization2FeaturesKHR sync2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
        .pNext = &dynamic_rendering,
        .synchronization2 = VK_TRUE,
    };

    VkPhysicalDeviceVulkan12Features vk12_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &sync2,
        .descriptorIndexing = VK_TRUE,
        .descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE,
        .descriptorBindingPartiallyBound = VK_TRUE,
        .runtimeDescriptorArray = VK_TRUE,
        .timelineSemaphore = VK_TRUE,
        .bufferDeviceAddress = VK_TRUE,
    };

    VkPhysicalDeviceFeatures basic_features{
        .shaderStorageImageReadWithoutFormat = VK_TRUE,
        .shaderStorageImageWriteWithoutFormat = VK_TRUE,
        .shaderInt64 = VK_TRUE,
    };

    VkDeviceCreateInfo device_info{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &vk12_features,
        .queueCreateInfoCount = static_cast<uint32_t>( queue_infos.size() ),
        .pQueueCreateInfos = queue_infos.data(),
        .enabledExtensionCount = static_cast<uint32_t>( settings.device_extensions.size() ),
        .ppEnabledExtensionNames = settings.device_extensions.data(),
        .pEnabledFeatures = &basic_features,
    };

    const auto result = vkCreateDevice( physical_device.physical_device, &device_info, nullptr, &device.device_ );
    if ( result != VK_SUCCESS ) {
        log_write( LogLevel::Error, "Failed to create a vulkan logical device, error returned: {:s}", result );
    }
    return result;
}

void Device::gather_queues( Device& device ) {
    const auto& physical_device = device.physical_devices_.front();
    const auto& queue_families = physical_device.queue_families;

    for ( uint32_t i = 0; i < queue_families.size(); ++i ) {
        for ( uint32_t j = 0; j < queue_families[i].queueCount; ++j ) {
            auto& wrapper = device.queues_.emplace_back( Queue{
                .queue = VK_NULL_HANDLE,
                .properties = queue_families[i],
                .family_index = i,
            } );

            vkGetDeviceQueue( device.device(), i, j, &wrapper.queue );
        }
    }
}

VkResult Device::create_allocator( Device& device ) {
    VmaVulkanFunctions vulkan_functions{};
    vulkan_functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkan_functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    vulkan_functions.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
    vulkan_functions.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
    vulkan_functions.vkAllocateMemory = vkAllocateMemory;
    vulkan_functions.vkFreeMemory = vkFreeMemory;
    vulkan_functions.vkMapMemory = vkMapMemory;
    vulkan_functions.vkUnmapMemory = vkUnmapMemory;
    vulkan_functions.vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges;
    vulkan_functions.vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges;
    vulkan_functions.vkBindBufferMemory = vkBindBufferMemory;
    vulkan_functions.vkBindImageMemory = vkBindImageMemory;
    vulkan_functions.vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements;
    vulkan_functions.vkGetImageMemoryRequirements = vkGetImageMemoryRequirements;
    vulkan_functions.vkCreateBuffer = vkCreateBuffer;
    vulkan_functions.vkDestroyBuffer = vkDestroyBuffer;
    vulkan_functions.vkCreateImage = vkCreateImage;
    vulkan_functions.vkDestroyImage = vkDestroyImage;
    vulkan_functions.vkCmdCopyBuffer = vkCmdCopyBuffer;
    vulkan_functions.vkGetBufferMemoryRequirements2KHR = vkGetBufferMemoryRequirements2KHR;
    vulkan_functions.vkGetImageMemoryRequirements2KHR = vkGetImageMemoryRequirements2KHR;
    vulkan_functions.vkBindBufferMemory2KHR = vkBindBufferMemory2KHR;
    vulkan_functions.vkBindImageMemory2KHR = vkBindImageMemory2KHR;
    vulkan_functions.vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2KHR;
    vulkan_functions.vkGetDeviceBufferMemoryRequirements = vkGetDeviceBufferMemoryRequirements;
    vulkan_functions.vkGetDeviceImageMemoryRequirements = vkGetDeviceImageMemoryRequirements;

    VmaAllocatorCreateInfo allocator_info = {
        .physicalDevice = device.physical_device(),
        .device = device.device(),
        .pVulkanFunctions = static_cast<const VmaVulkanFunctions*>( &vulkan_functions ),
        .instance = device.instance(),
        .vulkanApiVersion = VK_API_VERSION_1_0,
    };

    return vmaCreateAllocator( &allocator_info, &device.allocator_ );
}

constexpr LogLevel to_log_level( VkDebugUtilsMessageSeverityFlagBitsEXT severity ) {
    switch ( severity ) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT: return LogLevel::Trace;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT: return LogLevel::Info;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT: return LogLevel::Warn;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT: return LogLevel::Error;
        default: return LogLevel::None;
    }
}

constexpr std::string_view to_string( VkDebugUtilsMessageTypeFlagsEXT type ) {
    if ( type & VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT ) return "General";
    if ( type & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT ) return "Validation";
    if ( type & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT ) return "Performance";
    return "UNKNOWN";
}

VkBool32 Device::debug_callback( VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                                 VkDebugUtilsMessageTypeFlagsEXT message_type,
                                 const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
                                 void* ) {
    auto& debug_info = Device::debug_info_;

    switch ( message_severity ) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT: ++debug_info.num_verbose; break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT: ++debug_info.num_error; break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT: ++debug_info.num_warning; break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT: ++debug_info.num_info; break;
        default: ++debug_info.num_unknown; break;
    }

    log_write( to_log_level( message_severity ),
               "[Validation Layer - {}]: {}",
               to_string( message_type ),
               callback_data ? callback_data->pMessage : "unknown error" );

    return VK_FALSE;
}


Device::~Device() {
    resource_manager_.reset();
    pipeline_manager_.reset();
    swapchain_.reset();

    if ( allocator_ != VK_NULL_HANDLE ) {
        vmaCalculateStatistics( allocator_, &debug_info_.memory_stats_ );
        vmaDestroyAllocator( allocator_ );
    }

    if ( device_ != VK_NULL_HANDLE ) { vkDestroyDevice( device_, nullptr ); }

    if ( debug_messenger_ != VK_NULL_HANDLE ) {
        vkDestroyDebugUtilsMessengerEXT( instance_, debug_messenger_, nullptr );
    }

    if ( instance_ != VK_NULL_HANDLE ) { vkDestroyInstance( instance_, nullptr ); }
}


}// namespace aloe