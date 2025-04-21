#pragma once

#include <cassert>
#include <cstdint>

namespace aloe {

#pragma pack( push, 1 )
struct ResourceId {
    constexpr static uint32_t SLOT_BITS = 24;
    constexpr static uint32_t VERSION_BITS = 16;
    constexpr static uint32_t RESOURCE_ID_BITS = 24;

    ResourceId() : raw_( 0 ) {}

    ResourceId( uint64_t slot, uint64_t version, uint64_t id ) {
        assert( id > 0 );
        assert( slot < ( 1 << SLOT_BITS ) );
        assert( version < ( 1 << VERSION_BITS ) );
        assert( id < ( 1 << RESOURCE_ID_BITS ) );

        raw_ = slot + ( version << SLOT_BITS ) + ( id << ( SLOT_BITS + VERSION_BITS ) );
    }

    uint64_t raw() const { return raw_; }
    uint64_t slot() const { return raw_ & ( ( 1 << SLOT_BITS ) - 1 ); }
    uint64_t version() const { return ( raw_ >> SLOT_BITS ) & ( ( 1 << VERSION_BITS ) - 1 ); }
    uint64_t id() const { return ( raw_ >> ( SLOT_BITS + VERSION_BITS ) ) & ( ( 1 << RESOURCE_ID_BITS ) - 1 ); }

    auto operator<=>( const ResourceId& other ) const = default;

private:
    uint64_t raw_;
};

#pragma pack( pop )

static_assert( sizeof( ResourceId ) == sizeof( uint64_t ) );


struct BufferHandle : ResourceId {
    using ResourceId::ResourceId;
    auto operator<=>( const BufferHandle& other ) const = default;
};

struct ImageHandle : ResourceId {
    using ResourceId::ResourceId;
    auto operator<=>( const ImageHandle& other ) const = default;
};


}// namespace aloe

template<>
struct std::hash<aloe::BufferHandle> {
    size_t operator()( const aloe::BufferHandle& handle ) const noexcept {
        return std::hash<uint64_t>{}( handle.raw() );
    }
};

template<>
struct std::hash<aloe::ImageHandle> {
    size_t operator()( const aloe::ImageHandle& handle ) const noexcept {
        return std::hash<uint64_t>{}( handle.raw() );
    }
};