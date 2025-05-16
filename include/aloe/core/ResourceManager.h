#pragma once

#include <aloe/core/Handles.h>

#include <vma/vma.h>
#include <volk.h>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace aloe {
class Device;
class PipelineManager;

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
    friend class PipelineManager;

    struct DescriptorSlotAllocator {
        struct PendingWrite {
            std::variant<VkDescriptorBufferInfo, VkDescriptorImageInfo> resource;
            VkWriteDescriptorSet write;

            void finalize( VkDescriptorSet set );
        };

        DescriptorSlotAllocator( VkDescriptorType type, std::size_t max_slots );

        // Returns {slot, version} or nullopt if no slots are available
        std::optional<std::pair<uint32_t, uint32_t>>
        allocate_slot( const std::variant<VkDescriptorBufferInfo, VkDescriptorImageInfo>& resource );

        // Frees a slot and stages a null descriptor write
        void free_slot( uint32_t slot );

        // Version tracking
        uint32_t get_slot_version( uint32_t slot ) const;
        bool validate_slot( uint32_t slot, uint32_t version ) const;

        // Apply all pending writes to a descriptor set
        void bind_slots( VkDevice device, VkDescriptorSet set );

    private:
        const VkDescriptorType type_;
        const uint32_t max_slots_;

        std::vector<uint32_t> free_slots_;
        std::vector<uint32_t> versions_;
        std::vector<PendingWrite> pending_writes_;
    };

    template<typename ResourceT, typename ResourceDescT>
    struct AllocatedResource {
        struct BoundResource {
            VkImageView view = VK_NULL_HANDLE;// VK_NULL_HANDLE if `AllocatedResource` contains a buffer
            uint32_t slot = 0;                // Descriptor slot
            uint32_t version = 0;             // Slot version for validation
        };

        ResourceT resource = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        ResourceDescT desc = {};

        std::map<ResourceUsage, BoundResource> bound_resources = {};
    };

    Device& device_;
    VmaAllocator allocator_;

    uint32_t current_resource_id_ = 1;
    DescriptorSlotAllocator storage_buffer_allocator_;
    DescriptorSlotAllocator storage_image_allocator_;

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

    // Makes a resource binding for `usage` and returns the slot for the resource.
    std::optional<uint64_t> bind_resource( ResourceUsage usage );

    // Returns the number of bytes written
    VkDeviceSize upload_to_buffer( BufferHandle handle, const void* data, VkDeviceSize size );
    VkDeviceSize read_from_buffer( BufferHandle handle, void* out_data, VkDeviceSize bytes_to_read );

    VkDeviceSize upload_to_image( ImageHandle handle, const void* data, VkDeviceSize size );
    VkDeviceSize read_from_image( ImageHandle handle, void* out_data, VkDeviceSize bytes_to_read );

    VkBuffer get_buffer( BufferHandle handle ) const;
    VkImage get_image( ImageHandle handle ) const;
    VkImageView get_image_view( const ResourceUsage& usage ) const;

    void free_buffer( BufferHandle handle );
    void free_image( ImageHandle handle );

private:
    explicit ResourceManager( Device& device );

    // Helper to find and validate buffer before accessing it
    const AllocatedResource<VkBuffer, BufferDesc>* find_buffer( BufferHandle handle ) const;
    const AllocatedResource<VkImage, ImageDesc>* find_image( ImageHandle handle ) const;

    std::optional<uint64_t> bind_buffer( BufferHandle handle, const ResourceUsage& usage );
    std::optional<uint64_t> bind_image( ImageHandle handle, const ResourceUsage& usage );

    VkImageView create_view( ImageHandle handle, const ResourceUsage& usage ) const;

protected:// Internal API(s) for "friend"s to invoke.
    // Returns `true` if the resource(s) described by `usage` is valid
    bool validate_access( ResourceUsage usage );

    void bind_descriptors( VkDescriptorSet descriptor_set );
};

}// namespace aloe
