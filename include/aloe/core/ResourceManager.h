#pragma once

#include <aloe/core/Resources.h>

#include <vma/vma.h>
#include <volk.h>

#include <cstdint>
#include <string_view>
#include <unordered_map>

namespace aloe {
class Device;

struct BufferDesc {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO;
    VmaAllocationCreateFlags memory_flags = 0;
    const char* name = {};
};

struct ImageDesc {
    VkExtent3D extent = {};
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
    VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO;
    VmaAllocationCreateFlags memory_flags = 0;
    uint32_t mip_levels = 1;
    const char* name = {};
};

class ResourceManager {
    friend class Device;

    template<typename ResourceT, typename ResourceDescT>
    struct AllocatedResource {
        ResourceT resource = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        ResourceDescT desc = {};
    };

    Device& device_;
    VmaAllocator allocator_;

    uint64_t current_buffer_slot_ = 0;
    uint64_t current_image_slot_ = 0;
    uint64_t current_resource_id_ = 1;

    std::unordered_map<BufferHandle, AllocatedResource<VkBuffer, BufferDesc>> buffers_;
    std::unordered_map<ImageHandle, AllocatedResource<VkImage, ImageDesc>> images_;

public:
    ~ResourceManager();

    ResourceManager( ResourceManager& ) = delete;
    ResourceManager& operator=( const ResourceManager& other ) = delete;

    ResourceManager( ResourceManager&& ) = delete;
    ResourceManager& operator=( ResourceManager&& other ) = delete;

    BufferHandle create_buffer( const BufferDesc& desc );
    ImageHandle create_image( const ImageDesc& desc );

    // Returns the number of bytes written
    VkDeviceSize upload_to_buffer( BufferHandle handle, const void* data, VkDeviceSize size );
    VkDeviceSize read_from_buffer( BufferHandle handle, void* out_data, VkDeviceSize bytes_to_read );

    VkBuffer get_buffer( BufferHandle handle );
    VkImage get_image( ImageHandle handle );

    void free_buffer( BufferHandle handle );
    void free_image( ImageHandle handle );

private:
    explicit ResourceManager( Device& device );
};

}// namespace aloe