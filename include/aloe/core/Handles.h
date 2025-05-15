#pragma once

#include <cassert>
#include <cstdint>
#include <optional>
#include <variant>
#include <volk.h>

namespace aloe {

struct ResourceId {
    ResourceId( uint64_t raw ) : raw( raw ) {}
    ResourceId() : raw( 0 ) {}

    uint64_t raw;

    operator uint64_t() const noexcept { return raw; }
    auto operator<=>( const ResourceId& other ) const = default;
};

struct BufferHandle : ResourceId {
    using ResourceId::ResourceId;
    auto operator<=>( const BufferHandle& other ) const = default;
};

struct ImageHandle : ResourceId {
    using ResourceId::ResourceId;
    auto operator<=>( const ImageHandle& other ) const = default;
};

struct PipelineHandle {
    uint64_t id = 0;

    uint64_t operator()() const { return id; }
    auto operator<=>( const PipelineHandle& other ) const = default;
};

template<typename T>
struct ShaderUniform {
    static_assert( std::is_standard_layout_v<T> );
    explicit ShaderUniform( PipelineHandle pipeline, uint32_t offset ) : pipeline( pipeline ), offset( offset ) {}

    PipelineHandle pipeline;
    uint32_t offset;
    std::optional<T> data;

    ShaderUniform set_value( const T& value ) {
        data = value;
        return *this;
    }

    auto operator<=>( const ShaderUniform& ) const = default;
};

enum ResourceUsageKind {
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

struct ResourceUsage {
    std::variant<BufferHandle, ImageHandle> resource;
    VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_NONE_KHR;
    VkAccessFlags2 access = VK_ACCESS_2_NONE_KHR;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D;
    uint32_t base_mip_level = 0;
    uint32_t mip_count = 1;
    uint32_t base_array_layer = 0;
    uint32_t layer_count = 1;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;

    auto operator<=>( const ResourceUsage& ) const = default;

    template<typename ResourceT>
    static constexpr ResourceUsage make( ResourceT resource, ResourceUsageKind usage ) {
        static_assert( std::is_same_v<ResourceT, BufferHandle> || std::is_same_v<ResourceT, ImageHandle>,
                       "Resource must be either a BufferHandle or an ImageHandle." );

        ResourceUsage desc{};
        desc.resource = resource;

        switch ( usage ) {
            case ComputeStorageRead:
                desc.stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
                desc.access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR;
                desc.layout = VK_IMAGE_LAYOUT_GENERAL;
                break;

            case ComputeStorageWrite:
                desc.stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
                desc.access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
                desc.layout = VK_IMAGE_LAYOUT_GENERAL;
                break;

            case ComputeStorageReadWrite:
                desc.stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
                desc.access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
                desc.layout = VK_IMAGE_LAYOUT_GENERAL;
                break;

            case ComputeSampledRead:
            case FragmentSampledRead:
            case VertexShaderSampledRead:
                desc.stages = ( usage == ComputeSampledRead ) ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR
                    : ( usage == FragmentSampledRead )        ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT_KHR
                                                              : VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT_KHR;
                desc.access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT_KHR;
                desc.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                break;

            case FragmentStorageRead:
                desc.stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT_KHR;
                desc.access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR;
                desc.layout = VK_IMAGE_LAYOUT_GENERAL;
                break;

            case VertexBufferRead:
                desc.stages = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT_KHR;
                desc.access = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT_KHR;
                break;

            case IndexBufferRead:
                desc.stages = VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT_KHR;
                desc.access = VK_ACCESS_2_INDEX_READ_BIT_KHR;
                break;

            case ColorAttachmentWrite:
                desc.stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR;
                desc.access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT_KHR;
                desc.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                break;

            case ColorAttachmentReadWrite:
                desc.stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR;
                desc.access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT_KHR | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT_KHR;
                desc.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                break;

            case DepthStencilAttachmentWrite:
                desc.stages =
                    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT_KHR | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT_KHR;
                desc.access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT_KHR;
                desc.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                desc.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
                break;

            case DepthStencilAttachmentRead:
                desc.stages =
                    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT_KHR | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT_KHR;
                desc.access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT_KHR;
                desc.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                desc.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
                break;

            case TransferSrc:
                desc.stages = VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR;
                desc.access = VK_ACCESS_2_TRANSFER_READ_BIT_KHR;
                desc.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                break;

            case TransferDst:
                desc.stages = VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR;
                desc.access = VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR;
                desc.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                break;

            case Present:
                desc.stages = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT_KHR;
                desc.access = VK_ACCESS_2_NONE_KHR;
                desc.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                break;

            case Undefined:
            default:
                desc.stages = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT_KHR;
                desc.access = VK_ACCESS_2_NONE_KHR;
                desc.layout = VK_IMAGE_LAYOUT_UNDEFINED;
                break;
        }
        return desc;
    }
};

template<typename ResourceT>
constexpr ResourceUsage usage(ResourceT resource, ResourceUsageKind usage) {
     return ResourceUsage::make(resource, usage);
}

}// namespace aloe

template<>
struct std::hash<aloe::BufferHandle> {
    size_t operator()( const aloe::BufferHandle& handle ) const noexcept { return std::hash<uint64_t>{}( handle ); }
};

template<>
struct std::hash<aloe::ImageHandle> {
    size_t operator()( const aloe::ImageHandle& handle ) const noexcept { return std::hash<uint64_t>{}( handle ); }
};

template<>
struct std::hash<aloe::PipelineHandle> {
    size_t operator()( const aloe::PipelineHandle& handle ) const noexcept { return std::hash<uint64_t>{}( handle() ); }
};