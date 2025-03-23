#include <aloe/core/Swapchain.h>
#include <aloe/util/log.h>

#include <GLFW/glfw3.h>

namespace aloe {


tl::expected<std::unique_ptr<Swapchain>, VkResult> Swapchain::create_swapchain( const Device& ) {
    // Set our error callback before we do any glfw function calls in the off chance they fail.
    glfwSetErrorCallback( glfw_error_callback );

    auto swapchain = std::unique_ptr<Swapchain>( new Swapchain() );

    auto result = create_window( *swapchain );
    if ( result != VK_SUCCESS ) { return tl::make_unexpected( result ); }


    return swapchain;
}

Swapchain::~Swapchain() {
    if (window_ != nullptr) glfwDestroyWindow( window_);
}

VkResult Swapchain::create_window( Swapchain& swapchain ) {
    if (glfwInit() == GLFW_FALSE) return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    glfwWindowHint( GLFW_CLIENT_API, GLFW_NO_API );

    swapchain.window_ = glfwCreateWindow( 1920, 1080, "Window Title", nullptr, nullptr );
    if (swapchain.window_ == nullptr) return VkResult::VK_ERROR_INITIALIZATION_FAILED;

    return VK_SUCCESS;
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

    log_write(LogLevel::Error, "[GLFW] - {}: {}", to_readable_error(), message );
}

} // namespace aloe