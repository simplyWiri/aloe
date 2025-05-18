cmake_minimum_required (VERSION 3.30)
project ("aloe" CXX C)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# -Wno-missing-field-initializers: Vulkan has giant structs which we initialize with { }
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-missing-designated-field-initializers")

option(ALOE_ENABLE_TESTS "Enable or disable building the tests" OFF)

add_subdirectory(externals)
add_subdirectory(src)

if(ALOE_ENABLE_TESTS)
    message(STATUS "[aloe] Enabling unit tests")

    enable_testing()
    add_subdirectory(test)
endif()
