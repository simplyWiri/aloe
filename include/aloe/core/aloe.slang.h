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
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return 1;
        default: return -1;
    }
}

inline std::string aloe_shader_template() {
    return R"(
module aloe;

[[vk::binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0)]]
public RWByteAddressBuffer g_buffers[];

[[vk::binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0)]]
public RWTexture2D g_storage_images[];

namespace aloe {

// bottom 32 bits of the id we write to the descriptor is the slot index
constexpr static int64_t SLOT_INDEX_MASK = (1 << 32) - 1;

public struct BufferHandle {
    private uint64_t id;
    public RWByteAddressBuffer get() { return g_buffers[int(id & SLOT_INDEX_MASK)]; }
};

public struct ImageHandle {
    private uint64_t id;
    public RWTexture2D get() { return g_storage_images[int(id & SLOT_INDEX_MASK)]; }
};

}

)";
}

inline static std::string get_aloe_module() {
    auto source = aloe_shader_template();

    auto replace = [&]( std::string_view name, std::string_view value ) {
        auto pos = source.find( name );
        assert( pos != std::string::npos );

        source.replace( pos, name.length(), value );
    };

    replace( "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
             std::to_string( get_binding_slot( VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ) ) );
    replace( "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE",
             std::to_string( get_binding_slot( VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ) ) );

    return source;
}

}// namespace aloe
