# Dummy test executable
add_executable(basic_test core/device_tests.cpp core/swapchain_tests.cpp)
target_link_libraries(basic_test gtest gtest_main aloe)

# Add the test to CTest
add_test(NAME BasicTest COMMAND basic_test)