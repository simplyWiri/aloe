#include <aloe/core/Device.h>
#include <aloe/core/ResourceManager.h>
#include <aloe/util/log.h>

#include <algorithm>
#include <cassert>

namespace aloe {

ResourceManager::ResourceManager( Device& device ) : device_( device ), allocator_( device.allocator() ) {
    buffer_slot_versions_.resize( device_.get_physical_device_limits().maxDescriptorSetStorageBuffers, 0 );

    log_write( LogLevel::Info,
               "Device supports binding {} storage buffers in a single descriptor",
               buffer_slot_versions_.size() );
}

ResourceManager::~ResourceManager() {
    std::ranges::for_each( buffers_, [&]( const auto& pair ) {
        vmaDestroyBuffer( allocator_, pair.second.resource, pair.second.allocation );
    } );
    std::ranges::for_each( images_, [&]( const auto& pair ) {
        vmaDestroyImage( allocator_, pair.second.resource, pair.second.allocation );
    } );
}

BufferHandle ResourceManager::create_buffer( const BufferDesc& desc ) {
    VmaAllocationCreateInfo alloc_info{
        .flags = desc.memory_flags,
        .usage = desc.memory_usage,
    };

    AllocatedResource<VkBuffer, BufferDesc> buffer;
    VkBufferCreateInfo buffer_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = desc.size,
        .usage = desc.usage,
    };

    buffer.desc = desc;
    const auto result =
        vmaCreateBuffer( allocator_, &buffer_info, &alloc_info, &buffer.resource, &buffer.allocation, nullptr );
    if ( result != VK_SUCCESS ) { return {}; }

    if ( device_.validation_enabled() && desc.name ) {
        VkDebugUtilsObjectNameInfoEXT debug_name_info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .objectType = VK_OBJECT_TYPE_BUFFER,
            .objectHandle = reinterpret_cast<uint64_t>( buffer.resource ),
            .pObjectName = desc.name,
        };
        vkSetDebugUtilsObjectNameEXT( device_.device(), &debug_name_info );
    }

    // Get current slot and increment version
    const auto slot_index = current_buffer_slot_++;
    const auto current_version = ++buffer_slot_versions_[slot_index];

    return buffers_.emplace( BufferHandle( slot_index, current_version, current_resource_id_++ ), buffer ).first->first;
}

ImageHandle ResourceManager::create_image( const ImageDesc& desc ) {
    VmaAllocationCreateInfo alloc_info{
        .flags = desc.memory_flags,
        .usage = desc.memory_usage,
    };

    AllocatedResource<VkImage, ImageDesc> image;
    VkImageCreateInfo image_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = desc.format,
        .extent = desc.extent,
        .mipLevels = desc.mip_levels,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = desc.tiling,
        .usage = desc.usage,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    image.desc = desc;
    const auto result =
        vmaCreateImage( allocator_, &image_info, &alloc_info, &image.resource, &image.allocation, nullptr );
    if ( result != VK_SUCCESS ) { return {}; }

    if ( device_.validation_enabled() && desc.name ) {
        VkDebugUtilsObjectNameInfoEXT debug_name_info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .objectType = VK_OBJECT_TYPE_IMAGE,
            .objectHandle = reinterpret_cast<uint64_t>( image.resource ),
            .pObjectName = desc.name,
        };
        vkSetDebugUtilsObjectNameEXT( device_.device(), &debug_name_info );
    }

    return images_.emplace( ImageHandle( current_image_slot_++, 1, current_resource_id_++ ), image ).first->first;
}

VkDeviceSize ResourceManager::upload_to_buffer( BufferHandle handle, const void* data, VkDeviceSize size ) {
    if ( const auto* resource = find_buffer( handle ) ) {
        if ( ( resource->desc.memory_flags &
               ( VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT ) ) == 0 ) {
            log_write( LogLevel::Error,
                       "Trying to write to {}, which was not created with "
                       "`VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT` or "
                       "`VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT`",
                       resource->desc.name );

            return 0;
        }
        void* dst_pointer = nullptr;
        vmaMapMemory( allocator_, resource->allocation, &dst_pointer );
        std::memcpy( dst_pointer, data, size );
        vmaUnmapMemory( allocator_, resource->allocation );

        return size;
    }
    return 0;
}

VkDeviceSize ResourceManager::read_from_buffer( BufferHandle handle, void* out_data, VkDeviceSize bytes_to_read ) {
    if ( const auto* resource = find_buffer( handle ) ) {
        if ( ( resource->desc.memory_flags &
               ( VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT ) ) == 0 ) {
            log_write( LogLevel::Error,
                       "Trying to write to {}, which was not created with "
                       "`VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT` or "
                       "`VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT`",
                       resource->desc.name );

            return 0;
        }

        const auto read_bytes = std::min( resource->desc.size, bytes_to_read );

        void* dst_pointer = nullptr;
        vmaMapMemory( allocator_, resource->allocation, &dst_pointer );
        std::memcpy( out_data, dst_pointer, read_bytes );
        vmaUnmapMemory( allocator_, resource->allocation );

        return read_bytes;
    }
    return 0;
}

const ResourceManager::AllocatedResource<VkBuffer, BufferDesc>* ResourceManager::find_buffer( BufferHandle handle ) {
    auto report_error = [&]( std::string_view error_msg ) -> const AllocatedResource<VkBuffer, BufferDesc>* {
        log_write( LogLevel::Error,
                   "Invalid buffer handle {:#x} (resource_id: {}): {}",
                   handle.raw(),
                   handle.id(),
                   error_msg );
        return nullptr;
    };

    // Validate handle
    if ( handle.id() == 0 ) { return report_error( "Invalid buffer handle: resource ID is 0" ); }

    // Validate slot
    if ( handle.slot() >= buffer_slot_versions_.size() ) {
        return report_error( "Invalid buffer handle: slot out of range" );
    }

    // Validate version
    if ( handle.version() != buffer_slot_versions_[handle.slot()] ) {
        return report_error( std::format( "Invalid buffer handle: version mismatch (expected {}, got {})",
                                          buffer_slot_versions_[handle.slot()],
                                          handle.version() ) );
    }

    // Validate buffer exists
    const auto iter = buffers_.find( handle );
    if ( iter == buffers_.end() ) {
        return report_error( "Invalid buffer handle: buffer not found in active buffers" );
    }

    return &iter->second;
}

VkBuffer ResourceManager::get_buffer( BufferHandle handle ) {
    const auto* resource = find_buffer( handle );
    return resource ? resource->resource : VK_NULL_HANDLE;
}

VkImage ResourceManager::get_image( ImageHandle handle ) {
    const auto iter = images_.find( handle );
    return iter == images_.end() ? VK_NULL_HANDLE : iter->second.resource;
}

void ResourceManager::free_buffer( BufferHandle handle ) {
    const auto iter = buffers_.find( handle );
    assert( iter != buffers_.end() );
    if ( iter != buffers_.end() ) {
        vmaDestroyBuffer( allocator_, iter->second.resource, iter->second.allocation );

        // todo: pool "free" slots, instead of simple linear allocator approach
        if ( const auto slot = iter->first.slot(); ( slot + 1 ) == current_buffer_slot_ ) { --current_buffer_slot_; }

        buffers_.erase( iter );
    }
}

void ResourceManager::free_image( ImageHandle handle ) {
    const auto iter = images_.find( handle );
    assert( iter != images_.end() );
    if ( iter != images_.end() ) {
        vmaDestroyImage( allocator_, iter->second.resource, iter->second.allocation );
        images_.erase( iter );

        // const auto slot = iter->first.slot();
        // todo: mark `slot` as free
    }
}

}// namespace aloe