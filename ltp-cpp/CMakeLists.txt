if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif ()

project(ltp4cpp)
cmake_minimum_required(VERSION 3.15)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} cmake)
set(CMAKE_CXX_STANDARD 17)

option(ENABLE_LTO "Enable cross language linking time optimization" OFF)
if (ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT supported OUTPUT error)
    if (supported)
        message(STATUS "IPO / LTO enabled")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        add_link_options(-fuse-ld=lld)
    else ()
        message(STATUS "IPO / LTO not supported: <${error}>")
    endif ()
endif ()

find_package(onnxruntime REQUIRED)

include_directories(${CMAKE_BINARY_DIR}/rust)

add_subdirectory(rust)
add_executable(example cpp/example.cpp)
target_include_directories(example PUBLIC ${onnxruntime_INCLUDE_DIRS})
target_link_libraries(example ltp ${onnxruntime_LIBRARIES})