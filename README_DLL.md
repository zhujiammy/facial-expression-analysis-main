# 面部表情分析 DLL 使用说明

## 概述

本项目将面部表情分析功能打包为Windows DLL，可供C#、C++、Python等语言调用。基于ONNX模型、dlib人脸检测、OpenCV图像处理，实现高精度的情绪分析。

## 功能特性

- **高精度情绪分析**: 基于68个面部关键点的深度学习模型
- **多种输入格式**: 支持文件路径或内存字节流输入
- **完整情绪维度**: 返回arousal（唤醒度）、valence（愉悦度）、intensity（强度）
- **情绪分类**: 基于Russell情绪圆模型的详细情绪标签
- **跨语言调用**: 提供标准C接口，支持多种编程语言
- **线程安全**: 支持多线程环境使用

## 文件结构

```
facial-expression-analysis-main/
├── cpp_test/                    # C++ 源代码
│   ├── include/
│   │   ├── facial_expression_dll.h    # DLL接口头文件
│   │   └── emotion_analyzer.h          # 核心分析器头文件
│   ├── src/
│   │   ├── facial_expression_dll.cpp   # DLL接口实现
│   │   └── emotion_analyzer.cpp        # 核心分析器实现
│   └── CMakeLists.txt                  # CMake构建文件
├── csharp_test/                 # C# 测试代码
│   ├── FacialExpressionAPI.cs          # C# P/Invoke声明
│   ├── Program.cs                      # 测试程序
│   └── FacialExpressionTest.csproj     # 项目文件
├── models/                      # 模型文件
│   ├── model_emotion_pls30.onnx        # 情绪分析ONNX模型
│   ├── model_frontalization.npy        # 面部正面化模型
│   └── shape_predictor_68_face_landmarks.dat  # dlib面部关键点检测器
├── data/images/                 # 测试图片
├── build_all.bat               # 一键构建脚本
└── run_test.bat                # 测试运行脚本
```

## 快速开始

### 1. 环境要求

- Windows 10/11 (x64)
- Visual Studio 2019/2022
- CMake 3.16+
- .NET 6.0+ (用于C#测试)

### 依赖库
- OpenCV 4.x
- dlib
- ONNX Runtime
- cnpy (用于.npy文件读取)

### 2. 构建项目

```batch
# 一键构建DLL和C#测试程序
build_all.bat
```

### 3. 运行测试

```batch
# 运行C#测试程序
run_test.bat
```

## API 接口

### C++ 接口 (facial_expression_dll.h)

```cpp
// 情绪分析结果结构体
typedef struct {
    float arousal;        // 唤醒度 (-1.0 到 1.0)
    float valence;        // 愉悦度 (-1.0 到 1.0)
    float intensity;      // 强度 (0.0 到 1.0)
    char emotion_name[128];  // 情绪名称
    int success;          // 成功标志 (1=成功, 0=失败)
    char error_message[256]; // 错误信息
} EmotionResultDLL;

// 初始化情绪分析器
int InitializeEmotionAnalyzer(
    const char* onnx_model_path,
    const char* shape_predictor_path,
    const char* frontalization_model_path
);

// 从文件分析情绪
EmotionResultDLL AnalyzeEmotionFromFile(const char* image_path);

// 从字节数组分析情绪
EmotionResultDLL AnalyzeEmotionFromBytes(
    const unsigned char* image_data,
    int data_length,
    int width,  // 0表示自动解码
    int height, // 0表示自动解码
    int channels // 0表示自动解码
);

// 释放资源
void ReleaseEmotionAnalyzer();

// 获取最后的错误信息
const char* GetLastError();
```

### C# 接口示例

```csharp
using FacialExpressionAnalysis;

// 初始化
int result = FacialExpressionAPI.InitializeEmotionAnalyzer(
    "model_emotion_pls30.onnx",
    "shape_predictor_68_face_landmarks.dat",
    "model_frontalization.npy"
);

if (result == 1)
{
    // 分析图片文件
    EmotionResult emotion = FacialExpressionAPI.AnalyzeEmotionFromFile("test.jpg");
    
    if (emotion.Success == 1)
    {
        Console.WriteLine($"Arousal: {emotion.Arousal}");
        Console.WriteLine($"Valence: {emotion.Valence}");
        Console.WriteLine($"Intensity: {emotion.Intensity}");
        Console.WriteLine($"Emotion: {emotion.EmotionName}");
    }
    
    // 释放资源
    FacialExpressionAPI.ReleaseEmotionAnalyzer();
}
```

## 情绪分析结果说明

### 数值含义

- **Arousal (唤醒度)**: 范围 -1.0 到 1.0
  - 负值表示低唤醒（平静、困倦）
  - 正值表示高唤醒（兴奋、愤怒）

- **Valence (愉悦度)**: 范围 -1.0 到 1.0
  - 负值表示负面情绪（悲伤、愤怒）
  - 正值表示正面情绪（快乐、满足）

- **Intensity (强度)**: 范围 0.0 到 1.0
  - 表示情绪的强烈程度

### 情绪分类标签

基于Russell情绪圆模型，可能的情绪标签包括：

- **正面高唤醒**: "Very excited", "Excited", "Slightly excited"
- **正面低唤醒**: "Very pleased", "Pleased", "Slightly pleased"
- **负面高唤醒**: "Very annoyed", "Annoyed", "Slightly annoyed"
- **负面低唤醒**: "Very sad", "Sad", "Slightly sad"
- **中性**: "Neutral"

## 使用注意事项

### 1. 模型文件
- 确保所有模型文件与DLL在同一目录或指定正确路径
- 模型文件较大（~100MB），不要频繁移动

### 2. 图片要求
- 支持常见格式：JPG, PNG, BMP, TIFF等
- 图片中应包含清晰可见的人脸
- 建议图片分辨率不小于200x200像素

### 3. 内存管理
- C++: 使用完毕后调用 `ReleaseEmotionAnalyzer()`
- C#: 结构体会自动释放，但仍建议调用释放函数

### 4. 多线程使用
- 当前实现使用全局分析器对象
- 多线程环境下建议加锁或为每个线程创建独立实例

### 5. 错误处理
- 检查返回值的 `success` 字段
- 使用 `GetLastError()` 获取详细错误信息

## 性能优化建议

1. **初始化一次**: 避免重复初始化分析器
2. **批量处理**: 连续分析多张图片时保持分析器活跃
3. **图片预处理**: 预先裁剪包含人脸的区域可提高处理速度
4. **内存复用**: 重复使用相同尺寸的图片缓冲区

## 故障排除

### 常见错误

1. **"Emotion analyzer not initialized"**
   - 解决：确保先调用 `InitializeEmotionAnalyzer` 且返回成功

2. **"Failed to load model"**
   - 解决：检查模型文件路径和权限

3. **"No face detected"**
   - 解决：确保图片中包含清晰的人脸

4. **"ONNX Runtime not available"**
   - 解决：安装ONNX Runtime或检查DLL依赖

### 调试建议

1. 启用详细日志输出
2. 检查依赖库版本兼容性
3. 验证模型文件完整性
4. 测试简单的标准图片

## 更新日志

### v1.0.0
- 初始版本发布
- 支持基础情绪分析功能
- 提供C++和C#接口
- 完整的Russell情绪圆模型实现

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或联系开发团队。
