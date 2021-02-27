# LTP -- libltp

本仓库用于创建不同语言的绑定。

+ [x] Rust
+ [x] C++
+ [x] Java
+ [x] Python

## C++

```shell
cd ltp-cpp
cargo build --release
mkdir build && cd build
cmake ../example && make
```

### todo
+ [ ] use native class instead of arrow

## Java

```shell
cd ltp-java
make
gradle build
```

## Python

```shell
cd ltp-py
maturin build -i python3
```
