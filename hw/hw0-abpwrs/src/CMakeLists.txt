cmake_minimum_required(VERSION 3.10.2)
cmake_policy(VERSION 3.10.2...3.13.2)
project(HW0
        VERSION 0.0.1
        LANGUAGES CXX)

find_package(ITK 5.0 REQUIRED)

include(${ITK_USE_FILE})

add_executable(HW0 HW0.cxx)
target_link_libraries(HW0 ${ITK_LIBRARIES})