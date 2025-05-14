#pragma once

#include <aloe/core/handles.h>

#include <volk.h>

#include <cstdint>

namespace aloe {

struct BufferDesc {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO;
    VmaAllocationCreateFlags memory_flags = 0;
};

struct ImageDesc {
    VkExtent3D extent = {};
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
    VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO;
    VmaAllocationCreateFlags memory_flags = 0;
    uint32_t mip_levels = 1;
};

struct ResourceManager {
    ResourceManager( Device& device );
    ~ResourceManager();

    BufferHandle create_buffer( const char* name, const BufferDesc& desc );
    ImageHandle create_image( const char* name, const ImageDesc& desc );

    // Returns the number of bytes written
    VkDeviceSize upload_to_buffer( BufferHandle handle, const void* data, VkDeviceSize size );
    VkDeviceSize read_from_buffer( BufferHandle handle, void* out_data, VkDeviceSize bytes_to_read );

    VkDeviceSize upload_to_image( ImageHandle handle, const void* data, VkDeviceSize size );
    VkDeviceSize read_from_image( ImageHandle handle, void* out_data, VkDeviceSize bytes_to_read );

    void free_buffer( BufferHandle handle );
    void free_image( ImageHandle handle );
};

}// namespace aloe