#pragma once

#include <algorithm>
#include <ranges>
#include <unordered_set>
#include <vector>

#include "log.h"

namespace aloe {

template<typename Node>
concept HasGetDependents = requires( Node node ) {
    { node->get_dependents() } -> std::ranges::range;
};

template<std::ranges::range Range, typename Node = std::ranges::range_value_t<Range>>
    requires HasGetDependents<Node>
std::vector<Node> topological_sort( const Range& nodes ) {
    std::unordered_set<Node> visited;
    std::unordered_set<Node> recursion_stack;// Detect cycles
    std::vector<Node> sorted_order;
    bool has_cycle = false;

    auto dfs = [&]( Node node, auto&& dfs_ref ) -> void {
        if ( recursion_stack.contains( node ) ) {
            has_cycle = true;// Cycle detected
            return;
        }
        if ( visited.contains( node ) ) return;

        visited.insert( node );
        recursion_stack.insert( node );

        for ( const auto& neighbor : node->get_dependents() ) { dfs_ref( neighbor, dfs_ref ); }

        recursion_stack.erase( node );
        sorted_order.push_back( node );
    };

    for ( const auto& node : nodes ) {
        if ( !visited.contains( node ) ) { dfs( node, dfs ); }

        if ( has_cycle ) {
            assert( false );
            return {};
        }
    }


    std::reverse( sorted_order.begin(), sorted_order.end() );
    return sorted_order;
}


};// namespace aloe