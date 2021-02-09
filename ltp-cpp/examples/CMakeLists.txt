cmake_minimum_required(VERSION 3.17)
project(ltp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} cmake)

if (APPLE)
    set(OPENSSL_ROOT_DIR "/usr/local/opt/openssl@1.1/")
endif ()

add_executable(example example.cpp)

find_package(onnxruntime REQUIRED)
find_package(Arrow REQUIRED)


target_include_directories(example PUBLIC .. ${onnxruntime_INCLUDE_DIRS})
target_link_directories(example PUBLIC ../../target/release)
target_link_libraries(
        example PRIVATE
        ltp arrow_shared ${onnxruntime_LIBRARIES}
)