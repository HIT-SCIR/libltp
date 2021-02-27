if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_CMD cargo build --verbose)
    set(TARGET_DIR "debug")
else ()
    set(CARGO_CMD cargo build --release --verbose)
    set(TARGET_DIR "release")
endif ()

if (ENABLE_LTO)
    if (APPLE)
        set(RUSTFLAGS "-Clinker-plugin-lto")
    else ()
        set(RUSTFLAGS "-Clinker-plugin-lto" "-Clinker=clang" "-Clink-arg=-fuse-ld=lld")
    endif ()
endif ()

execute_process(
        OUTPUT_VARIABLE CARGO_PROJECT_TOP
        COMMAND cargo locate-project --workspace --message-format=plain
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

execute_process(
        OUTPUT_VARIABLE CARGO_PROJECT_TOP
        COMMAND dirname ${CARGO_PROJECT_TOP}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

string(STRIP "${CARGO_PROJECT_TOP}" CARGO_PROJECT_TOP)

set(RUST_PART_LIB "${CARGO_PROJECT_TOP}/target/${TARGET_DIR}/libltp.a")
set(RUST_PART_CXX "${CMAKE_CURRENT_BINARY_DIR}/ltp.cpp")
set(RUST_PART_HED "${CMAKE_CURRENT_BINARY_DIR}/ltp.h")

add_library(ltp STATIC ${RUST_PART_CXX} ${RUST_PART_HED})
add_custom_command(
        OUTPUT ${RUST_PART_CXX} ${RUST_PART_HED} ${RUST_PART_LIB}
        COMMAND ${CMAKE_COMMAND} -E env RUSTFLAGS=${RUST_FLAGS} ${CARGO_CMD}
        COMMAND cp ${CARGO_PROJECT_TOP}/target/cxxbridge/ltp4cpp/src/lib.rs.h ${RUST_PART_HED}
        COMMAND cp ${CARGO_PROJECT_TOP}/target/cxxbridge/ltp4cpp/src/lib.rs.cc ${RUST_PART_CXX}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ltp pthread dl ${RUST_PART_LIB})

add_test(
        NAME ltp
        COMMAND cargo test
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
