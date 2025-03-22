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
endfunction()

aloe_add_library(aloe
    HEADERS
        core/Device.h
    SOURCES
        core/Device.cpp
        LINK_AGAINST
        volk
        tl::expected
)