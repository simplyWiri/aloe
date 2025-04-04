#include <aloe/core/PipelineManager.h>
#include <aloe/util/log.h>

#include <slang.h>

#include <filesystem>
#include <iostream>
#include <ranges>

namespace aloe {

PipelineManager::PipelineManager( Device& device, std::vector<std::string> root_paths )
    : device_( device )
    , root_paths_( std::move( root_paths ) ) {
}

tl::expected<PipelineHandle, std::string> PipelineManager::compile_pipeline( const ComputePipelineInfo& ) {
    return tl::make_unexpected( "Not implemented error" );
}

void PipelineManager::set_define( const std::string&, const std::string& ) {
}

void PipelineManager::set_virtual_file( const std::string&, const std::string& ) {
}

uint64_t PipelineManager::get_pipeline_version( PipelineHandle ) const {
    return 0;
}

const std::vector<uint32_t>& PipelineManager::get_pipeline_spirv( PipelineHandle ) const {
    static std::vector<uint32_t> empty;
    return empty;
}

}// namespace aloe
