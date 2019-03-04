cmake_minimum_required(VERSION 3.10.2)
cmake_policy(VERSION 3.10.2...3.13.2)

#-----------------------------------------------------------------------------
# Enable C++11 by default, allow C++14 or C++17
##  Set the default target properties for ITK
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11) # Supported values are ``11``, ``14``, and ``17``.
endif()
if(NOT CMAKE_CXX_STANDARD_REQUIRED)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(NOT CMAKE_CXX_EXTENSIONS)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()
set(VALID_CXX_STANDARDS "11" "14" "17")
if(NOT CMAKE_CXX_STANDARD IN_LIST VALID_CXX_STANDARDS )
    MESSAGE(FATAL_ERROR "CMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD} not in know standards list\n ${VALID_CXX_STANDARDS}  for BRAINSTools version 5 and greater.")
endif()
#-----------------------------------------------------------------------------

project( HW2
        VERSION 0.0.1
        LANGUAGES CXX)

find_package(ITK 5.0 REQUIRED)
include(${ITK_USE_FILE})

add_subdirectory(src)