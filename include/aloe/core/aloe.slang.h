#pragma once

#include <cassert>
#include <format>
#include <string>
#include <string_view>

#define VK_ENABLE_BETA_EXTENSIONS
#include <volk.h>

namespace aloe {

constexpr static int get_binding_slot( VkDescriptorType type ) {
    switch ( type ) {
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: return 0;
        default: return -1;
    }
}

std::string aloe_shader_template() {
    return R"(
module aloe;

[[vk::binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0)]]
public RWByteAddressBuffer g_buffers[];

)";
}

static std::string get_aloe_module() {
    auto source = aloe_shader_template();

    auto replace = [&]( std::string_view name, std::string_view value ) {
        auto pos = source.find( name );
        assert( pos != std::string::npos );

        source.replace( pos, name.length(), value );
    };

    replace( "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
             std::to_string( get_binding_slot( VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ) ) );

    return source;
}

}// namespace aloe
