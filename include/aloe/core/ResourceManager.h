#pragma once

#include <vma/vma.h>
#include <volk.h>

#include <cstdint>
#include <string_view>

namespace aloe {

struct BufferDesc {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO;
    std::string_view debug_name = {};
};

struct BufferHandle {
    const uint64_t id = 0;

    uint64_t operator()() const { return id; }
    auto operator<=>( const BufferHandle& other ) const = default;
};

struct ImageDesc {
    VkExtent3D extent = {};
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
    VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO;
    uint32_t mip_levels = 1;
    std::string_view debug_name = {};
};

struct ImageHandle {
    const uint64_t id = 0;

    uint64_t operator()() const { return id; }
    auto operator<=>( const ImageHandle& other ) const = default;
};

struct ResourceManager {
    explicit ResourceManager( VmaAllocator ) {}

    BufferHandle create_buffer( const BufferDesc& ) { return { 0 }; }
    ImageHandle create_image( const ImageDesc& ) { return { 0 }; }

    VkBuffer get_buffer( BufferHandle ) { return VK_NULL_HANDLE; }
    VkImage get_image( ImageHandle ) { return VK_NULL_HANDLE; }
};

}// namespace aloe