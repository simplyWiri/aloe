#include <aloe/core/Device.h>
#include <aloe/core/ResourceManager.h>
#include <aloe/util/log.h>

#include <algorithm>
#include <cassert>
#include <numeric>

#include "aloe/core/aloe.slang.h"

namespace aloe {

ResourceManager::ResourceManager( Device& device )
    : device_( device )
    , allocator_( device.allocator() )
    , storage_buffer_allocator_( VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 device.get_physical_device_limits().maxDescriptorSetStorageBuffers )
    , storage_image_allocator_( VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                device.get_physical_device_limits().maxDescriptorSetStorageImages ) {
}

void ResourceManager::DescriptorSlotAllocator::PendingWrite::finalize( VkDescriptorSet set ) {
    if ( std::holds_alternative<VkDescriptorBufferInfo>( resource ) ) {
        write.pBufferInfo = &std::get<VkDescriptorBufferInfo>( resource );
    } else if ( std::holds_alternative<VkDescriptorImageInfo>( resource ) ) {
        write.pImageInfo = &std::get<VkDescriptorImageInfo>( resource );
    }
    write.dstSet = set;
}

ResourceManager::DescriptorSlotAllocator::DescriptorSlotAllocator( VkDescriptorType type, std::size_t max_slots )
    : type_( type )
    , max_slots_( static_cast<uint32_t>( max_slots ) ) {
    free_slots_.resize( max_slots_ );
    versions_.resize( max_slots_, 0 );
    std::iota( free_slots_.begin(), free_slots_.end(), 0 );
}

std::optional<std::pair<uint32_t, uint32_t>> ResourceManager::DescriptorSlotAllocator::allocate_slot(
    const std::variant<VkDescriptorBufferInfo, VkDescriptorImageInfo>& resource ) {
    if ( free_slots_.empty() ) return std::nullopt;

    const uint32_t slot = free_slots_.back();
    free_slots_.pop_back();
    ++versions_[slot];

    auto& pending = pending_writes_.emplace_back();
    pending.resource = resource;

    auto& write = pending.write;
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.descriptorCount = 1;
    write.dstArrayElement = slot;
    write.descriptorType = type_;
    write.dstBinding = get_binding_slot( type_ );

    return std::make_pair( slot, versions_[slot] );
}

void ResourceManager::DescriptorSlotAllocator::free_slot( uint32_t slot ) {
    if ( slot >= max_slots_ ) return;
    if ( std::ranges::find( free_slots_, slot ) != free_slots_.end() ) return;

    free_slots_.emplace_back( slot );
}

uint32_t ResourceManager::DescriptorSlotAllocator::get_slot_version( uint32_t slot ) const {
    return slot < versions_.size() ? versions_[slot] : 0;
}

bool ResourceManager::DescriptorSlotAllocator::validate_slot( uint32_t slot, uint32_t version ) const {
    return slot < versions_.size() && versions_[slot] == version;
}

void ResourceManager::DescriptorSlotAllocator::bind_slots( VkDevice device, VkDescriptorSet set ) {
    if ( pending_writes_.empty() ) return;

    std::vector<VkWriteDescriptorSet> writes;
    for ( auto& pending : pending_writes_ ) {
        pending.finalize( set );
        writes.emplace_back( pending.write );
    }

    vkUpdateDescriptorSets( device, static_cast<uint32_t>( writes.size() ), writes.data(), 0, nullptr );

    pending_writes_.clear();
}


ResourceManager::~ResourceManager() {
    std::ranges::for_each( buffers_, [&]( const auto& pair ) {
        vmaDestroyBuffer( allocator_, pair.second.resource, pair.second.allocation );
    } );

    std::ranges::for_each( images_, [&]( const auto& pair ) {
        std::ranges::for_each( pair.second.bound_resources, [&]( const auto& bound_resource ) {
            assert( bound_resource.second.view != VK_NULL_HANDLE );
            vkDestroyImageView( device_.device(), bound_resource.second.view, nullptr );
        } );

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

    return buffers_.emplace( BufferHandle( current_resource_id_++ ), buffer ).first->first;
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

    return images_.emplace( current_resource_id_++, image ).first->first;
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

VkDeviceSize ResourceManager::upload_to_image( ImageHandle handle, const void* data, VkDeviceSize size ) {
    if ( const auto* resource = find_image( handle ) ) {
        // Create staging buffer
        BufferHandle staging_buffer =
            create_buffer( { .size = size,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             .memory_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                             .name = "Image Upload Staging Buffer" } );

        // Copy data to staging buffer
        upload_to_buffer( staging_buffer, data, size );

        // Get transfer queue
        const auto transfer_queues = device_.queues_by_capability( VK_QUEUE_TRANSFER_BIT );
        if ( transfer_queues.empty() ) {
            log_write( LogLevel::Error, "No transfer queue available for image upload" );
            free_buffer( staging_buffer );
            return 0;
        }

        device_.immediate_submit( transfer_queues[0], [&]( VkCommandBuffer cmd ) {
            // Transition to transfer dst
            VkImageMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                          .srcAccessMask = 0,
                                          .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                          .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                          .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                          .image = resource->resource,
                                          .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                                .baseMipLevel = 0,
                                                                .levelCount = resource->desc.mip_levels,
                                                                .baseArrayLayer = 0,
                                                                .layerCount = 1 } };

            vkCmdPipelineBarrier( cmd,
                                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr,
                                  1,
                                  &barrier );

            // Copy buffer to image
            VkBufferImageCopy region{ .bufferOffset = 0,
                                      .bufferRowLength = 0,
                                      .bufferImageHeight = 0,
                                      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                            .mipLevel = 0,
                                                            .baseArrayLayer = 0,
                                                            .layerCount = 1 },
                                      .imageOffset = { 0, 0, 0 },
                                      .imageExtent = resource->desc.extent };

            vkCmdCopyBufferToImage( cmd,
                                    get_buffer( staging_buffer ),
                                    resource->resource,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    1,
                                    &region );

            // Transition to shader read
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

            vkCmdPipelineBarrier( cmd,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                  0,
                                  0,
                                  nullptr,
                                  0,
                                  nullptr,
                                  1,
                                  &barrier );
        } );

        free_buffer( staging_buffer );
        return size;
    }
    return 0;
}

VkDeviceSize ResourceManager::read_from_image( ImageHandle handle, void* out_data, VkDeviceSize bytes_to_read ) {
    if ( const auto* resource = find_image( handle ) ) {
        // Create staging buffer
        BufferHandle staging_buffer = create_buffer( {
            .size = bytes_to_read,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .memory_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .name = "Image Download Staging Buffer",
        } );

        const auto transfer_queues = device_.queues_by_capability( VK_QUEUE_TRANSFER_BIT );
        if ( transfer_queues.empty() ) {
            log_write( LogLevel::Error, "No transfer queue available for image download" );
            free_buffer( staging_buffer );
            return 0;
        }

        device_.immediate_submit( transfer_queues[0], [&]( VkCommandBuffer cmd ) {
            // Copy directly from GENERAL layout
            VkBufferImageCopy region{
                .bufferOffset = 0,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                      .mipLevel = 0,
                                      .baseArrayLayer = 0,
                                      .layerCount = 1 },
                .imageOffset = { 0, 0, 0 },
                .imageExtent = resource->desc.extent,
            };

            vkCmdCopyImageToBuffer( cmd,
                                    resource->resource,
                                    VK_IMAGE_LAYOUT_GENERAL,
                                    get_buffer( staging_buffer ),
                                    1,
                                    &region );
        } );

        vkDeviceWaitIdle( device_.device() );

        // Read from staging buffer
        read_from_buffer( staging_buffer, out_data, bytes_to_read );
        free_buffer( staging_buffer );

        return bytes_to_read;
    }
    return 0;
}

const ResourceManager::AllocatedResource<VkBuffer, BufferDesc>*
ResourceManager::find_buffer( BufferHandle handle ) const {
    auto report_error = [&]( std::string_view error_msg ) -> const AllocatedResource<VkBuffer, BufferDesc>* {
        log_write( LogLevel::Error, "Invalid buffer handle {}: {}", handle.raw, error_msg );
        return nullptr;
    };

    // Validate handle
    if ( handle.raw == 0 ) { return report_error( "Invalid buffer handle: resource ID is 0" ); }

    // Validate buffer exists
    const auto iter = buffers_.find( handle );
    if ( iter == buffers_.end() ) {
        return report_error( "Invalid buffer handle: buffer not found in active buffers" );
    }

    return &iter->second;
}

const ResourceManager::AllocatedResource<VkImage, ImageDesc>* ResourceManager::find_image( ImageHandle handle ) const {
    auto report_error = [&]( std::string_view error_msg ) -> const AllocatedResource<VkImage, ImageDesc>* {
        log_write( LogLevel::Error, "Invalid image handle {}: {}", handle.raw, error_msg );
        return nullptr;
    };

    // Validate handle
    if ( handle.raw == 0 ) { return report_error( "Invalid image handle: resource ID is 0" ); }

    // Validate image exists
    const auto iter = images_.find( handle );
    if ( iter == images_.end() ) { return report_error( "Invalid image handle: image not found in active images" ); }

    return &iter->second;
}

VkImageView ResourceManager::create_view( ImageHandle handle, const ResourceUsage& usage ) const {
    auto* resource = find_image( handle );

    const VkImageViewCreateInfo view_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = resource->resource,
        .viewType = usage.view_type,
        .format = resource->desc.format,
        .components = {
            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
            .a = VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        .subresourceRange = {
            .aspectMask = usage.aspect,
            .baseMipLevel = usage.base_mip_level,
            .levelCount = usage.mip_count,
            .baseArrayLayer = usage.base_array_layer,
            .layerCount = usage.layer_count,
        }
    };

    VkImageView view = VK_NULL_HANDLE;
    const auto result = vkCreateImageView( device_.device(), &view_info, nullptr, &view );
    return result != VK_SUCCESS ? VK_NULL_HANDLE : view;
}

bool ResourceManager::validate_access( ResourceUsage usage ) {
    return std::visit(
        [&]( const auto& resource ) {
            using T = std::decay_t<decltype( resource )>;

            if constexpr ( std::is_same_v<T, BufferHandle> ) {
                const auto* buffer = find_buffer( resource );
                return buffer != nullptr && buffer->bound_resources.contains( usage );
            }
            if constexpr ( std::is_same_v<T, ImageHandle> ) {
                const auto* image = find_image( resource );
                return image != nullptr && image->bound_resources.contains( usage );
            }

            assert( false );
        },
        usage.resource );
}

VkBuffer ResourceManager::get_buffer( BufferHandle handle ) const {
    const auto* resource = find_buffer( handle );
    return resource ? resource->resource : VK_NULL_HANDLE;
}

VkImage ResourceManager::get_image( ImageHandle handle ) const {
    const auto iter = images_.find( handle );
    return iter == images_.end() ? VK_NULL_HANDLE : iter->second.resource;
}

void ResourceManager::free_buffer( BufferHandle handle ) {
    const auto iter = buffers_.find( handle );
    assert( iter != buffers_.end() );
    if ( iter != buffers_.end() ) {
        vmaDestroyBuffer( allocator_, iter->second.resource, iter->second.allocation );

        for ( const auto& bound_resource : iter->second.bound_resources ) {
            storage_buffer_allocator_.free_slot( bound_resource.second.slot );
        }

        buffers_.erase( iter );
    }
}

void ResourceManager::free_image( ImageHandle handle ) {
    const auto iter = images_.find( handle );
    assert( iter != images_.end() );
    if ( iter != images_.end() ) {
        std::ranges::for_each( iter->second.bound_resources, [&]( const auto& bound_resource ) {
            vkDestroyImageView( device_.device(), bound_resource.second.view, nullptr );
        } );
        vmaDestroyImage( allocator_, iter->second.resource, iter->second.allocation );

        for ( const auto& bound_resource : iter->second.bound_resources ) {
            storage_buffer_allocator_.free_slot( bound_resource.second.slot );
        }

        images_.erase( iter );
    }
}
std::optional<uint64_t> ResourceManager::bind_buffer( BufferHandle handle, const ResourceUsage& usage ) {
    auto* resource = const_cast<AllocatedResource<VkBuffer, BufferDesc>*>( find_buffer( handle ) );
    if ( !resource ) return std::nullopt;

    // Check if we already have a valid binding
    if ( const auto binding_it = resource->bound_resources.find( usage );
         binding_it != resource->bound_resources.end() ) {
        const auto& bound_resource = binding_it->second;
        // Validate the slot is still valid
        if ( storage_buffer_allocator_.validate_slot( bound_resource.slot, bound_resource.version ) ) {
            return handle.raw >> 32 | bound_resource.slot;
        }
    }

    // Create a new descriptor binding
    VkDescriptorBufferInfo buffer_info{ .buffer = resource->resource, .offset = 0, .range = resource->desc.size };

    auto slot_version = storage_buffer_allocator_.allocate_slot( buffer_info );
    if ( !slot_version ) return std::nullopt;

    // Cache the binding
    resource->bound_resources[usage] = {
        .view = VK_NULL_HANDLE,// Buffers don't have views
        .slot = slot_version->first,
        .version = slot_version->second,
    };

    return handle.raw << 32 | slot_version->first;
}

std::optional<uint64_t> ResourceManager::bind_image( ImageHandle handle, const ResourceUsage& usage ) {
    auto* resource = const_cast<AllocatedResource<VkImage, ImageDesc>*>( find_image( handle ) );
    if ( !resource ) return std::nullopt;

    // Check if we already have a valid binding
    if ( const auto view_it = resource->bound_resources.find( usage ); view_it != resource->bound_resources.end() ) {
        const auto& bound_view = view_it->second;
        // Ensure the slot is still valid
        assert( storage_image_allocator_.validate_slot( bound_view.slot, bound_view.version ) );
        return handle.raw >> 32 | bound_view.slot;
    }

    // Create or reuse view
    const auto view = create_view( handle, usage );
    if ( view == VK_NULL_HANDLE ) return std::nullopt;

    // Bind to descriptor
    VkDescriptorImageInfo image_info{ .imageView = view, .imageLayout = usage.layout };

    auto slot_version = storage_image_allocator_.allocate_slot( image_info );
    if ( slot_version == std::nullopt ) {
        vkDestroyImageView( device_.device(), view, nullptr );
        return std::nullopt;
    }

    resource->bound_resources[usage] = {
        .view = view,
        .slot = slot_version->first,
        .version = slot_version->second,
    };

    return handle.raw << 32 | slot_version->first;
}

std::optional<uint64_t> ResourceManager::bind_resource( ResourceUsage usage ) {
    return std::visit(
        [&]( const auto& resource ) {
            using T = std::decay_t<decltype( resource )>;

            if constexpr ( std::is_same_v<T, BufferHandle> ) { return bind_buffer( resource, usage ); }
            if constexpr ( std::is_same_v<T, ImageHandle> ) { return bind_image( resource, usage ); }
            assert( false );
        },
        usage.resource );
}

void ResourceManager::bind_descriptors( VkDescriptorSet descriptor_set ) {
    storage_buffer_allocator_.bind_slots( device_.device(), descriptor_set );
    storage_image_allocator_.bind_slots( device_.device(), descriptor_set );
}

}// namespace aloe