#pragma once

#include <functional>
#include <vector>

#include <aloe/util/log.h>

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