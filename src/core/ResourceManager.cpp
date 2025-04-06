#include <aloe/core/Device.h>
#include <aloe/core/ResourceManager.h>

#include <algorithm>
#include <cassert>

namespace aloe {

ResourceManager::ResourceManager( Device& device ) : device_( device ), allocator_( device.allocator() ) {
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
        .usage = desc.memory_usage,
    };

    AllocatedResource<VkBuffer> buffer;
    VkBufferCreateInfo buffer_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = desc.size,
        .usage = desc.usage,
    };

    const auto result =
        vmaCreateBuffer( allocator_, &buffer_info, &alloc_info, &buffer.resource, &buffer.allocation, nullptr );
    if ( result != VK_SUCCESS ) { return { 0 }; }

    if ( device_.validation_enabled() && desc.name ) {
        VkDebugUtilsObjectNameInfoEXT debug_name_info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .objectType = VK_OBJECT_TYPE_BUFFER,
            .objectHandle = reinterpret_cast<uint64_t>( buffer.resource ),
            .pObjectName = desc.name,
        };
        vkSetDebugUtilsObjectNameEXT( device_.device(), &debug_name_info );
    }

    return buffers_.emplace( ++current_buffer_id_, buffer ).first->first;
}

ImageHandle ResourceManager::create_image( const ImageDesc& desc ) {
    VmaAllocationCreateInfo alloc_info{ .usage = desc.memory_usage };

    AllocatedResource<VkImage> image;
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

    const auto result =
        vmaCreateImage( allocator_, &image_info, &alloc_info, &image.resource, &image.allocation, nullptr );
    if ( result != VK_SUCCESS ) { return { 0 }; }

    if ( device_.validation_enabled() && desc.name ) {
        VkDebugUtilsObjectNameInfoEXT debug_name_info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .objectType = VK_OBJECT_TYPE_IMAGE,
            .objectHandle = reinterpret_cast<uint64_t>( image.resource ),
            .pObjectName = desc.name,
        };
        vkSetDebugUtilsObjectNameEXT( device_.device(), &debug_name_info );
    }

    return images_.emplace( ++current_image_id_, image ).first->first;
}

VkBuffer ResourceManager::get_buffer( BufferHandle handle ) {
    const auto iter = buffers_.find( handle );
    return iter == buffers_.end() ? VK_NULL_HANDLE : iter->second.resource;
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
        buffers_.erase( iter );
    }
}

void ResourceManager::free_image( ImageHandle handle ) {
    const auto iter = images_.find( handle );
    assert( iter != images_.end() );
    if ( iter != images_.end() ) {
        vmaDestroyImage( allocator_, iter->second.resource, iter->second.allocation );
        images_.erase( iter );
    }
}

}// namespace aloe