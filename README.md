# libltp<sup>beta</sup>

本仓库用于为 [Language Technology Platform](https://github.com/HIT-SCIR/ltp) 创建不同语言的绑定。

+ [x] Rust
+ [x] C++
+ [x] Java
+ [x] Python

## 模型地址

|   模型    |                      大小                       |                              SHA256                              |
| :-------: | :---------------------------------------------: | :--------------------------------------------------------------: |
| Base  | [491.9MB](http://39.96.43.154/libltp/base.tgz)  | 2df2575325e740e894ba865c74ef81498a5ffaabca58626e977dc5dc95cb2808 |
| Small | [156.8MB](http://39.96.43.154/libltp/small.tgz) | 85db4bcf6b1df8f0a0d44466050587229f294a4134c525c167942786052c292d |
| Tiny  |  [31.3MB](http://39.96.43.154/libltp/tiny.tgz)  | e662eae1ed4c904eeefe33534526bf05c007a44187be52e0f47b5c2f78b7e29d |

## C++

```shell
cd ltp-cpp
mkdir build && cd build
cmake -DENABLE_LTO=ON -DCMAKE_BUILD_TYPE=Release ..
make
```

## Java

```shell
cd ltp-java/rust && make lib
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

## 开源协议

1. 语言技术平台面向国内外大学、中科院各研究所以及个人研究者免费开放源代码，但如上述机构和个人将该平台用于商业目的（如企业合作项目等）则需要付费。
2. 除上述机构以外的企事业单位，如申请使用该平台，需付费。
3. 凡涉及付费问题，请发邮件到 car@ir.hit.edu.cn 洽商。
4. 如果您在 LTP 基础上发表论文或取得科研成果，请您在发表论文和申报成果时声明“使用了哈工大社会计算与信息检索研究中心研制的语言技术平台（LTP）”.
   同时，发信给car@ir.hit.edu.cn，说明发表论文或申报成果的题目、出处等。
