#pragma once

#include <vma/vma.h>
#include <volk.h>

#include <cstdint>
#include <string_view>
#include <unordered_map>

namespace aloe {
class Device;

struct BufferHandle {
    uint64_t id = 0;

    uint64_t operator()() const { return id; }
    auto operator<=>( const BufferHandle& other ) const = default;
};

struct ImageHandle {
    uint64_t id = 0;

    uint64_t operator()() const { return id; }
    auto operator<=>( const ImageHandle& other ) const = default;
};

}// namespace aloe

template<>
struct std::hash<aloe::BufferHandle> {
    size_t operator()( const aloe::BufferHandle& handle ) const noexcept { return std::hash<uint64_t>{}( handle.id ); }
};

template<>
struct std::hash<aloe::ImageHandle> {
    size_t operator()( const aloe::ImageHandle& handle ) const noexcept { return std::hash<uint64_t>{}( handle.id ); }
};

namespace aloe {

struct BufferDesc {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO;
    const char* name = {};
};

struct ImageDesc {
    VkExtent3D extent = {};
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
    VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO;
    uint32_t mip_levels = 1;
    const char* name = {};
};

class ResourceManager {
    template<typename ResourceT>
    struct AllocatedResource {
        ResourceT resource = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
    };

    Device& device_;
    VmaAllocator allocator_;

    uint64_t current_buffer_id_ = 0;
    uint64_t current_image_id_ = 0;

    std::unordered_map<BufferHandle, AllocatedResource<VkBuffer>> buffers_;
    std::unordered_map<ImageHandle, AllocatedResource<VkImage>> images_;

public:
    explicit ResourceManager( Device& device );
    ~ResourceManager();

    BufferHandle create_buffer( const BufferDesc& desc );
    ImageHandle create_image( const ImageDesc& desc );

    VkBuffer get_buffer( BufferHandle handle );
    VkImage get_image( ImageHandle handle );

    void free_buffer( BufferHandle handle );
    void free_image( ImageHandle handle );
};

}// namespace aloe