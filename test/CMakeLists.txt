# Dummy test executable
add_executable(core_tests
        core/command_list_tests.cpp
        core/device_tests.cpp
        core/resource_manager_tests.cpp
)

add_executable(window_tests
        core/swapchain_tests.cpp
)

add_executable(pipeline_tests
        core/pipeline_manager_tests.cpp
)

find_package(SPIRV-Tools CONFIG REQUIRED)

target_link_libraries(core_tests gtest gtest_main aloe)
target_link_libraries(window_tests gtest gtest_main aloe)
target_link_libraries(pipeline_tests gtest gtest_main aloe SPIRV-Tools-static)

# Add the test to CTest
add_test(NAME "Core Tests" COMMAND core_tests)
add_test(NAME "Window Tests" COMMAND window_tests)
add_test(NAME "Pipeline Tests" COMMAND pipeline_tests)

add_custom_command(TARGET pipeline_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/core/resources $<TARGET_FILE_DIR:pipeline_tests>/resources)
