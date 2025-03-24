#include <aloe/core/Device.h>
#include <aloe/core/Swapchain.h>
#include <aloe/util/log.h>
#include <aloe/util/vulkan_util.h>

#include <GLFW/glfw3.h>
#include <volk.h>

#include <algorithm>

namespace aloe {


tl::expected<std::unique_ptr<Swapchain>, VkResult> Swapchain::create_swapchain( const Device& device,
                                                                                SwapchainSettings settings ) {
    // Set our error callback before we do any glfw function calls in the off chance they fail.
    glfwSetErrorCallback( glfw_error_callback );

    auto swapchain = std::unique_ptr<Swapchain>( new Swapchain( device ) );
    swapchain->use_hdr_ = settings.use_hdr_surface;

    auto result = create_window( *swapchain, settings );
    if ( result != VK_SUCCESS ) { return tl::make_unexpected( result ); }

    result = create_surface( device, *swapchain );
    if ( result != VK_SUCCESS ) { return tl::make_unexpected( result ); }

    setup_callbacks( *swapchain );
    result = swapchain->load_surface_capabilities();
    if ( result != VK_SUCCESS ) { return tl::make_unexpected( result ); }

    result = swapchain->build_swapchain();
    if ( result != VK_SUCCESS ) { return tl::make_unexpected( result ); }

    return swapchain;
}

Swapchain::~Swapchain() {
    std::ranges::for_each( image_views_,
                           [&]( const auto& view ) { vkDestroyImageView( device_.device(), view, nullptr ); } );
    if ( swapchain_ != VK_NULL_HANDLE ) vkDestroySwapchainKHR( device_.device(), swapchain_, nullptr );
    if ( surface_ != VK_NULL_HANDLE ) vkDestroySurfaceKHR( device_.instance(), surface_, nullptr );
    if ( window_ != nullptr ) glfwDestroyWindow( window_ );

    glfwTerminate();
}

bool Swapchain::poll_events() {
    glfwPollEvents();

    return glfwWindowShouldClose( window_ );
}

void Swapchain::resize() {
    vkDeviceWaitIdle( device_.device() );

    // They will have changed, when we receive this callback
    auto result = load_surface_capabilities();
    if ( result != VK_SUCCESS ) {
        log_write( LogLevel::Error,
                   "Failed to reload surface capabilities following window resize, error: {}",
                   result );
    }

    result = build_swapchain();
    if ( result != VK_SUCCESS ) {
        log_write( LogLevel::Error, "Failed to rebuild swapchain following window resize, error: {}", result );
    }
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

    hdr_supported_ = std::ranges::any_of( formats_, [&]( const VkSurfaceFormatKHR& format ) {
        return std::tie( format.format, format.colorSpace ) == std::tie( hdr_target.format, hdr_target.colorSpace );
    } );

    auto sdr_supported = std::ranges::any_of( formats_, [&]( const VkSurfaceFormatKHR& format ) {
        return std::tie( format.format, format.colorSpace ) == std::tie( sdr_target.format, sdr_target.colorSpace );
    } );
    auto supports_fifo = std::ranges::find( present_modes_, VK_PRESENT_MODE_FIFO_KHR ) != present_modes_.end();

    log_write(
        LogLevel::Info,
        "Loaded surface capabilities. Framebuffer extent: {}x{}. HDR target found: {}. SDR target found: {}. FIFO "
        "present mode supported: {}. Min image count: {}, Max image count: {}.",
        capabilities_.currentExtent.width,
        capabilities_.currentExtent.height,
        hdr_supported_,
        sdr_supported,
        supports_fifo,
        capabilities_.minImageCount,
        capabilities_.maxImageCount );

    return supports_fifo && sdr_supported ? VK_SUCCESS : VK_ERROR_FORMAT_NOT_SUPPORTED;
}

VkResult Swapchain::build_swapchain() {
    vkDeviceWaitIdle( device_.device() );

    auto old_swapchain = swapchain_;

    // Selects the optimal surface format and present mode based on available options.
    auto [width, height] = capabilities_.currentExtent;
    auto surface_format = ( use_hdr_ && hdr_supported_ ) ? hdr_target : sdr_target;

    if ( width <= 0 || height <= 0 ) { return VK_ERROR_SURFACE_LOST_KHR; }

    VkSwapchainCreateInfoKHR swapchainCI = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface_,
        .minImageCount = capabilities_.minImageCount,
        .imageFormat = surface_format.format,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = { width, height },
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = VK_PRESENT_MODE_FIFO_KHR,
        .clipped = VK_TRUE,
        .oldSwapchain = old_swapchain,
    };

    auto result = vkCreateSwapchainKHR( device_.device(), &swapchainCI, nullptr, &swapchain_ );
    if ( result != VK_SUCCESS ) return result;

    if ( old_swapchain != VK_NULL_HANDLE ) {
        for ( const auto& view : image_views_ ) { vkDestroyImageView( device_.device(), view, nullptr ); }
        vkDestroySwapchainKHR( device_.device(), old_swapchain, nullptr );

        // The memory of the image itself is owned and managed by the swapchain, and does not need to be freed.
        image_views_.clear();
        images_.clear();
    }

    images_ = get_enumerated_value<VkImage>(
        [&]( uint32_t* c, VkImage* images ) { vkGetSwapchainImagesKHR( device_.device(), swapchain_, c, images ); },
        "Could not get swapchain images" );

    VkImageViewCreateInfo view = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = surface_format.format,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    for ( std::size_t i = 0; i < images_.size(); i++ ) {
        view.image = images_[i];

        auto& view_handle = image_views_.emplace_back();
        if ( vkCreateImageView( device_.device(), &view, nullptr, &view_handle ) != VK_SUCCESS ) return result;
    }

    return images_.size() ? VK_SUCCESS : VK_ERROR_FORMAT_NOT_SUPPORTED;
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

void Swapchain::window_resized( GLFWwindow* window, int, int ) {
    // Resize gets the size of the swapchain from the surface, which is updated by GLFW.
    static_cast<Swapchain*>( glfwGetWindowUserPointer( window ) )->resize();
}

Swapchain::Swapchain( const Device& device ) : device_( device ) {
}

}// namespace aloe