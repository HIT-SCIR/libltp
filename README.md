# LTP -- libltp

本仓库用于创建不同语言的绑定。

+ [x] Rust
+ [x] C++
+ [x] Java
+ [x] Python

## C++

```shell
cd ltp-cpp
mkdir build && cd build
cmake -DENABLE_LTO=ON -DCMAKE_BUILD_TYPE=Release ..
make
```

## Java

```shell
cd ltp-java/rust && make
cd .. && gradle build
```

## Python

```shell
cd ltp-py
maturin build -i python3
```

## TODO
+ [ ] 可选任务
+ [ ] 语言词语对齐
+ [x] 其他语言支持 cuda
+ [ ] 自动下载模型