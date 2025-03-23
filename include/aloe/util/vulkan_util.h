#pragma once

#include <aloe/util/log.h>

#include <volk/volk.h>

#include <functional>
#include <sstream>
#include <string>
#include <vector>

template<>
struct std::formatter<VkResult> : std::formatter<std::string> {
    auto format( VkResult result, format_context& ctx ) const {
        std::ostringstream o;
        switch ( result ) {
            case VK_SUCCESS: o << "VK_SUCCESS"; break;
            case VK_NOT_READY: o << "VK_NOT_READY"; break;
            case VK_TIMEOUT: o << "VK_TIMEOUT"; break;
            case VK_EVENT_SET: o << "VK_EVENT_SET"; break;
            case VK_EVENT_RESET: o << "VK_EVENT_RESET"; break;
            case VK_INCOMPLETE: o << "VK_INCOMPLETE"; break;
            case VK_ERROR_OUT_OF_HOST_MEMORY: o << "VK_ERROR_OUT_OF_HOST_MEMORY"; break;
            case VK_ERROR_OUT_OF_DEVICE_MEMORY: o << "VK_ERROR_OUT_OF_DEVICE_MEMORY"; break;
            case VK_ERROR_INITIALIZATION_FAILED: o << "VK_ERROR_INITIALIZATION_FAILED"; break;
            case VK_ERROR_DEVICE_LOST: o << "VK_ERROR_DEVICE_LOST"; break;
            case VK_ERROR_MEMORY_MAP_FAILED: o << "VK_ERROR_MEMORY_MAP_FAILED"; break;
            case VK_ERROR_LAYER_NOT_PRESENT: o << "VK_ERROR_LAYER_NOT_PRESENT"; break;
            case VK_ERROR_EXTENSION_NOT_PRESENT: o << "VK_ERROR_EXTENSION_NOT_PRESENT"; break;
            case VK_ERROR_FEATURE_NOT_PRESENT: o << "VK_ERROR_FEATURE_NOT_PRESENT"; break;
            case VK_ERROR_INCOMPATIBLE_DRIVER: o << "VK_ERROR_INCOMPATIBLE_DRIVER"; break;
            case VK_ERROR_TOO_MANY_OBJECTS: o << "VK_ERROR_TOO_MANY_OBJECTS"; break;
            case VK_ERROR_FORMAT_NOT_SUPPORTED: o << "VK_ERROR_FORMAT_NOT_SUPPORTED"; break;
            case VK_ERROR_FRAGMENTED_POOL: o << "VK_ERROR_FRAGMENTED_POOL"; break;
            case VK_ERROR_UNKNOWN: o << "VK_ERROR_UNKNOWN"; break;
            // Add other VkResult values here if needed
            default: o << "VK_RESULT_UNKNOWN"; break;
        }

        return formatter<string>::format(std::move(o).str(), ctx);
    }
};

namespace aloe {

    template<class T>
    std::vector<T> get_enumerated_value( std::function<void( uint32_t*, T* )> function, const char* error_message ) {
        uint32_t count = 0;
        function( &count, nullptr );
        if ( count == 0 ) {
            log_write( LogLevel::Error, "Failed to enumerate values, error: {}", error_message );
            return {};
        }
        std::vector<T> return_vector( count );
        function( &count, return_vector.data() );
        return return_vector;
    }

}// namespace aloe