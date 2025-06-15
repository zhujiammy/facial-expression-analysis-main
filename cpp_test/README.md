# 面部表情分析 - C++ ONNX 模型测试

这个项目提供了一个C++实现，用于测试ONNX格式的面部表情分析模型，并与原始的Python joblib模型进行比较验证。

## 项目结构

```
cpp_test/
├── CMakeLists.txt              # CMake构建文件
├── README.md                   # 项目说明文档
├── compare_with_cpp.py         # Python比较脚本
├── include/                    # 头文件目录
│   ├── emotion_analyzer.h      # 情感分析器
│   ├── facial_landmarks.h      # 面部关键点处理
│   ├── model_comparison.h      # 模型比较工具
│   └── utils.h                 # 工具函数
├── src/                        # 源代码目录
│   ├── main.cpp               # 主程序
│   ├── emotion_analyzer.cpp   # 情感分析器实现
│   ├── facial_landmarks.cpp   # 面部关键点处理实现
│   ├── model_comparison.cpp   # 模型比较工具实现
│   └── utils.cpp              # 工具函数实现
└── build/                      # 构建目录
```

## 依赖项

### 必需的库

1. **OpenCV** (>= 4.0)
   - 图像处理和计算机视觉
   
2. **dlib** (>= 19.20)
   - 面部检测和关键点定位
   
3. **ONNX Runtime** (>= 1.8)
   - ONNX模型推理引擎
   
4. **cnpy**
   - 读取NumPy .npy文件
   
5. **CMake** (>= 3.16)
   - 构建系统

### Windows 安装指南

#### 使用 vcpkg 安装依赖

```powershell
# 安装vcpkg（如果尚未安装）
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# 安装依赖项
.\vcpkg install opencv[contrib]:x64-windows
.\vcpkg install dlib:x64-windows
.\vcpkg install cnpy:x64-windows
```

#### 安装 ONNX Runtime

1. 从官网下载预编译版本：https://github.com/microsoft/onnxruntime/releases
2. 解压到 `C:\Program Files\onnxruntime`
3. 或者根据实际路径修改 CMakeLists.txt 中的路径

### Linux 安装指南

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install cmake build-essential

# 安装OpenCV
sudo apt-get install libopencv-dev

# 安装dlib
sudo apt-get install libdlib-dev

# 安装ONNX Runtime
# 下载并安装预编译版本或从源码编译

# 安装cnpy
git clone https://github.com/rogersce/cnpy.git
cd cnpy
mkdir build && cd build
cmake ..
make -j4
sudo make install
```

## 构建项目

### Windows (Visual Studio)

```powershell
cd cpp_test
mkdir build
cd build

# 使用vcpkg工具链
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake

# 构建
cmake --build . --config Release
```

### Linux

```bash
cd cpp_test
mkdir build && cd build
cmake ..
make -j4
```

## 使用方法

### 分析单张图像

```bash
# Windows
.\build\bin\Release\FacialExpressionAnalysis.exe -i ../data/images/pleased.jpg

# Linux
./build/bin/FacialExpressionAnalysis -i ../data/images/pleased.jpg
```

### 与Python模型比较

```bash
# Windows
.\build\bin\Release\FacialExpressionAnalysis.exe -c

# Linux
./build/bin/FacialExpressionAnalysis -c
```

### 随机输入一致性测试

```bash
# Windows
.\build\bin\Release\FacialExpressionAnalysis.exe -r

# Linux
./build/bin/FacialExpressionAnalysis -r
```

### 运行完整验证测试

```bash
# Windows
.\build\bin\Release\FacialExpressionAnalysis.exe -v

# Linux
./build/bin/FacialExpressionAnalysis -v
```

### 命令行选项

```
用法: FacialExpressionAnalysis [选项]
选项:
  -h, --help              显示帮助信息
  -i, --image <path>      分析单张图像
  -b, --batch <dir>       批量分析目录中的图像
  -c, --compare           与Python模型比较
  -r, --random-test       随机输入一致性测试
  -v, --validate          运行完整验证测试
  -m, --models <dir>      指定模型文件目录（默认: ../models）
```

## 模型文件

确保以下模型文件存在于 `../models/` 目录中：

1. `model_emotion_pls30.onnx` - ONNX格式的情感分析模型
2. `model_frontalization.npy` - 面部正面化模型
3. `shape_predictor_68_face_landmarks.dat` - dlib 68点面部关键点检测器

## 预期输出

### 单张图像分析

```
========== 分析单张图像 ==========
图像路径: ../data/images/pleased.jpg
图像尺寸: 640x480

检测到 68 个面部关键点

情感分析结果:
  Arousal (唤醒度): 0.123456
  Valence (效价): 0.654321
  Intensity (强度): 0.666667
  Emotion (情感): slightly pleased

结果图像已保存: result_2025-06-14_15-30-45.jpg
```

### 模型比较结果

```
========== 模型比较报告 ==========
时间: 2025-06-14 15:30:45

✅ 比较成功
测试样本数: 3
最大差异: 0.00000123
平均差异: 0.00000045

详细预测对比:
样本 1:
  C++:    [0.123456, 0.654321]
  Python: [0.123456, 0.654321]
  差异:   [0.000000, 0.000000]

🎉 结论: 模型预测高度一致，转换成功！
```

## 故障排除

### 常见问题

1. **模型文件找不到**
   - 确保模型文件路径正确
   - 使用 `-m` 选项指定正确的模型目录

2. **库链接错误**
   - 检查所有依赖库是否正确安装
   - 更新 CMakeLists.txt 中的库路径

3. **ONNX Runtime 错误**
   - 确保 ONNX Runtime 版本兼容
   - 检查模型文件是否有效

4. **面部检测失败**
   - 确保图像质量良好
   - 检查 dlib 模型文件是否存在

### 调试模式

编译调试版本：

```bash
# Debug 模式
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
```

## 性能基准

在典型的现代CPU上：
- 单张图像分析：~50-100ms
- ONNX模型推理：~1-5ms
- 面部关键点检测：~20-40ms

## 许可证

本项目遵循与原始Python项目相同的许可证。

## 贡献

欢迎提交问题报告和改进建议。

## 更新日志

### v1.0.0 (2025-06-14)
- 初始版本
- 支持ONNX模型推理
- 完整的模型一致性验证
- 面部关键点可视化
- 多平台支持（Windows/Linux）
