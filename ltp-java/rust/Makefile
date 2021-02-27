ifeq ($(shell uname),Darwin)
    EXT := dylib
else
    EXT := so
endif

WORKSPACE := $(shell cargo locate-project --workspace --message-format=plain)
ROOT_DIR = $(shell dirname ${WORKSPACE})
RELEASE_DIR = ${ROOT_DIR}/target/release
RESOURCE_DIR = ${ROOT_DIR}/ltp-java/src/main/resources


javac:
	javac -h . ${ROOT_DIR}/ltp-java/src/main/java/cn/edu/hit/ir/LTP.java

lib: src/lib.rs Cargo.toml
	cargo build --release
	mkdir -p ${RESOURCE_DIR}
	cp ${RELEASE_DIR}/libltp4j.$(EXT) ${RESOURCE_DIR}/libltp4j.$(EXT)

all: lib
