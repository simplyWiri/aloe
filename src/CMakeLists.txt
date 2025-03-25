# To make the actual definition of the library a little less - "scatter brained" we instead make a helper function to
# abstract all logic regarding exporting / library creation.
# Define a helper function to simplify adding libraries.
function(aloe_add_library libname)
    cmake_parse_arguments(ALC "" "" "SOURCES;HEADERS;LINK_AGAINST" ${ARGN})

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

aloe_add_library(aloe
    HEADERS
        core/Device.h
        core/PipelineManager.h
        core/Swapchain.h
    SOURCES
        core/Device.cpp
        core/PipelineManager.cpp
        core/Swapchain.cpp
        LINK_AGAINST
        glfw
        volk
        tl::expected
)