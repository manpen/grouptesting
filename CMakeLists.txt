cmake_minimum_required(VERSION 3.10)
project(grouptesting)

add_subdirectory(extlib/tlx)
add_subdirectory(extlib/range-v3)

set(CMAKE_CXX_STANDARD 17)
include_directories(include)
find_package(OpenMP)

add_executable(grouptesting main.cpp)
target_link_libraries(grouptesting tlx range-v3 OpenMP::OpenMP_CXX)

enable_testing()
add_subdirectory(tests)