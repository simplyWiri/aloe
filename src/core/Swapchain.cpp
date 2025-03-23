#include <algorithm>

#include <aloe/core/Device.h>
#include <aloe/core/Swapchain.h>
#include <aloe/util/log.h>
#include <aloe/util/vulkan_util.h>

#include <GLFW/glfw3.h>
#include <volk.h>

namespace aloe {


tl::expected<std::unique_ptr<Swapchain>, VkResult> Swapchain::create_swapchain( const Device& device,
                                                                                SwapchainSettings settings ) {
    // Set our error callback before we do any glfw function calls in the off chance they fail.
    glfwSetErrorCallback( glfw_error_callback );

    auto swapchain = std::unique_ptr<Swapchain>( new Swapchain( device ) );

    auto result = create_window( *swapchain, settings );
    if ( result != VK_SUCCESS ) { return tl::make_unexpected( result ); }

    result = create_surface( device, *swapchain );
    if ( result != VK_SUCCESS ) { return tl::make_unexpected( result ); }

    setup_callbacks( *swapchain );
    swapchain->load_surface_capabilities();

    return swapchain;
}

Swapchain::~Swapchain() {
    if ( surface_ != VK_NULL_HANDLE ) vkDestroySurfaceKHR( device_.instance(), surface_, nullptr );
    if ( window_ != nullptr ) glfwDestroyWindow( window_ );

    glfwTerminate();
}

void Swapchain::resize( int, int ) {
}

VkResult Swapchain::load_surface_capabilities() {
    const auto physical_device = device_.physical_device();

    const auto result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR( physical_device, surface_, &capabilities_ );
    if ( result != VK_SUCCESS ) return result;

    formats_ = get_enumerated_value<VkSurfaceFormatKHR>(
        [&]( uint32_t* count, VkSurfaceFormatKHR* data ) {
            return vkGetPhysicalDeviceSurfaceFormatsKHR( physical_device, surface_, count, data );
        },
        "Failed to enumerate surface formats" );
    present_modes_ = get_enumerated_value<VkPresentModeKHR>(
        [&]( uint32_t* count, VkPresentModeKHR* data ) {
            return vkGetPhysicalDeviceSurfacePresentModesKHR( physical_device, surface_, count, data );
        },
        "Failed to enumerate surface present modes" );

    log_write(
        LogLevel::Info,
        "Loaded surface capabilities. Framebuffer extent: {}x{}. HDR target found: {}. SDR target found: {}. FIFO "
        "present mode supported: {}. Min image count: {}, Max image count: {}.",
        capabilities_.currentExtent.width,
        capabilities_.currentExtent.height,
        std::ranges::any_of( formats_,
                             [&]( const VkSurfaceFormatKHR& format ) {
                                 return std::tie( format.format, format.colorSpace ) ==
                                     std::tie( hdr_target.format, hdr_target.colorSpace );
                             } ),
        std::ranges::any_of( formats_,
                             [&]( const VkSurfaceFormatKHR& format ) {
                                 return std::tie( format.format, format.colorSpace ) ==
                                     std::tie( sdr_target.format, sdr_target.colorSpace );
                             } ),
        std::ranges::find( present_modes_, VK_PRESENT_MODE_FIFO_KHR ) != present_modes_.end(),
        capabilities_.minImageCount,
        capabilities_.maxImageCount );

    return ( formats_.size() && present_modes_.size() ) ? VK_SUCCESS : VK_ERROR_FORMAT_NOT_SUPPORTED;
}

VkResult Swapchain::create_window( Swapchain& swapchain, const SwapchainSettings& settings ) {
    glfwWindowHint( GLFW_CLIENT_API, GLFW_NO_API );

    swapchain.window_ = glfwCreateWindow( settings.width, settings.height, settings.title, nullptr, nullptr );
    if ( swapchain.window_ == nullptr ) return VkResult::VK_ERROR_INITIALIZATION_FAILED;

    return VK_SUCCESS;
}

VkResult Swapchain::create_surface( const Device& device, Swapchain& swapchain ) {
    return glfwCreateWindowSurface( device.instance(), swapchain.window_, nullptr, &swapchain.surface_ );
}

void Swapchain::setup_callbacks( Swapchain& swapchain ) {
    // Allows us to use `glfwGetWindowUserPointer`, which returns `swapchain` in static functions which GLFW invokes
    glfwSetWindowUserPointer( swapchain.window_, &swapchain );

    glfwSetWindowSizeCallback( swapchain.window_, window_resized );
}

void Swapchain::glfw_error_callback( int error_code, const char* message ) {
    const auto to_readable_error = [=]() -> const char* {
        switch ( error_code ) {
            case GLFW_NOT_INITIALIZED: return "NOT_INITIALIZED";
            case GLFW_NO_CURRENT_CONTEXT: return "NO_CURRENT_CONTEXT";
            case GLFW_INVALID_ENUM: return "INVALID_ENUM";
            case GLFW_INVALID_VALUE: return "INVALID_VALUE";
            case GLFW_OUT_OF_MEMORY: return "OUT_OF_MEMORY";
            case GLFW_API_UNAVAILABLE: return "API_UNAVAILABLE";
            case GLFW_VERSION_UNAVAILABLE: return "VERSION_UNAVAILABLE";
            case GLFW_PLATFORM_ERROR: return "PLATFORM_ERROR";
            default: return "UNKNOWN ERROR";
        }
    };

    log_write( LogLevel::Error, "[GLFW] - {}: {}", to_readable_error(), message );
}

void Swapchain::window_resized( GLFWwindow* window, int width, int height ) {
    auto& swapchain = *static_cast<Swapchain*>( glfwGetWindowUserPointer( window ) );
    swapchain.resize( width, height );
}

Swapchain::Swapchain( const Device& device ) : device_( device ) {
}

}// namespace aloe