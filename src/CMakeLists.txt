# To make the actual definition of the library a little less - "scatter brained" we instead make a helper function to
# abstract all logic regarding exporting / library creation.
# Define a helper function to simplify adding libraries.
function(aloe_add_library libname)
    cmake_parse_arguments(ALC "" "" "SOURCES;LINK_AGAINST" ${ARGN})

    add_library(${libname} STATIC ${ALC_SOURCES})
    target_include_directories(${libname} PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
            $<INSTALL_INTERFACE:include>
    )
    if(ALC_LINK_AGAINST)
        target_link_libraries(${libname} PUBLIC ${ALC_LINK_AGAINST})
    endif()

    # Slang doesn't properly work when its brought in as a dependency via submodules (https://github.com/shader-slang/slang/issues/6676)
    target_link_libraries(${libname} PUBLIC $ENV{VULKAN_SDK}/Lib/libslang.dylib)
    target_include_directories(${libname} PUBLIC $ENV{VULKAN_SDK}/include/slang)
endfunction()

aloe_add_library(vma
    SOURCES
        vma.cpp
    LINK_AGAINST
        VulkanMemoryAllocator
        volk
)
target_compile_options(vma PRIVATE -w) # vma does not build cleanly.

aloe_add_library(aloe
    HEADERS
        core/CommandList.h
        core/Device.h
        core/PipelineManager.h
        core/ResourceManager.h
        core/Swapchain.h
        core/TaskGraph.cpp
SOURCES
        core/CommandList.cpp
        core/Device.cpp
        core/PipelineManager.cpp
        core/ResourceManager.cpp
        core/Swapchain.cpp
        core/TaskGraph.cpp
    LINK_AGAINST
        glfw
        vma
        volk
)