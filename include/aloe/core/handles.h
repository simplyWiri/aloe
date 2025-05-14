#pragma once

#include <vma/vma.h>

#include <compare>
#include <cstdint>
#include <optional>
#include <type_traits>

// Forward Declares our strongly typed resource handles, and forward declares classes.

namespace aloe {

struct CommandList;
struct Device;
struct FrameGraph;
struct PipelineManager;
struct ResourceManager;
struct Swapchain;
struct TaskGraph;

struct ImageHandle {
    int64_t value;
    auto operator<=>( const ImageHandle& ) const = default;
};
struct BufferHandle {
    int64_t value;
    auto operator<=>( const BufferHandle& ) const = default;
};
struct PipelineHandle {
    int64_t value;
    auto operator<=>( const PipelineHandle& ) const = default;
};

template<typename T>
    requires( std::is_standard_layout_v<T> )
struct ShaderUniform {
    PipelineHandle pipeline;
    uint32_t offset;
    std::optional<T> data;

    ShaderUniform& set_value( const T& value );
    auto operator<=>( const ShaderUniform& ) const = default;
};

struct ResourceUsageDesc {
    enum Usage {
        // Compute
        ComputeStorageRead,
        ComputeStorageWrite,
        ComputeStorageReadWrite,
        ComputeSampledRead,

        // Fragment
        FragmentSampledRead,
        FragmentStorageRead,

        // Vertex
        VertexBufferRead,
        IndexBufferRead,
        VertexShaderSampledRead,

        // Attachment
        ColorAttachmentWrite,
        ColorAttachmentReadWrite,
        DepthStencilAttachmentWrite,
        DepthStencilAttachmentRead,

        // Transfer
        TransferSrc,
        TransferDst,

        // Special
        Present,
        Undefined
    };

    union {
        BufferHandle buffer_handle;
        ImageHandle image_handle;
    };

    VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_NONE_KHR;
    VkAccessFlags2 access = VK_ACCESS_2_NONE_KHR;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D;
    uint32_t base_mip_level = 0;
    uint32_t mip_count = 1;
    uint32_t base_array_layer = 0;
    uint32_t layer_count = 1;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;

    auto operator<=>( const ResourceUsageDesc& ) const = default;

    static constexpr ResourceUsageDesc make( ImageHandle resource, Usage usage );
    static constexpr ResourceUsageDesc make( BufferHandle resource, Usage usage );
};
}// namespace aloe