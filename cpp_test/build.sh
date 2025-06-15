#!/bin/bash

echo "========================================"
echo "  面部表情分析 - C++ 项目构建脚本"
echo "========================================"

# 设置变量
BUILD_DIR="build"
CMAKE_BUILD_TYPE="Release"

# 检查是否在正确的目录
if [ ! -f "CMakeLists.txt" ]; then
    echo "错误: 请在包含 CMakeLists.txt 的目录中运行此脚本"
    exit 1
fi

# 检查必要的工具
command -v cmake >/dev/null 2>&1 || { echo "错误: 需要安装 cmake"; exit 1; }
command -v make >/dev/null 2>&1 || { echo "错误: 需要安装 make"; exit 1; }

# 创建构建目录
if [ ! -d "$BUILD_DIR" ]; then
    echo "创建构建目录: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# 运行 CMake 配置
echo ""
echo "配置项目..."
cmake .. -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
if [ $? -ne 0 ]; then
    echo "错误: CMake 配置失败"
    exit 1
fi

# 编译项目
echo ""
echo "编译项目..."
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo "错误: 编译失败"
    exit 1
fi

# 检查构建结果
if [ -f "bin/FacialExpressionAnalysis" ]; then
    echo ""
    echo "✅ 构建成功！"
    echo "可执行文件位置: $(pwd)/bin/FacialExpressionAnalysis"
    echo ""
    echo "使用示例:"
    echo "  $(pwd)/bin/FacialExpressionAnalysis -h"
    echo "  $(pwd)/bin/FacialExpressionAnalysis -i ../../data/images/pleased.jpg"
    echo "  $(pwd)/bin/FacialExpressionAnalysis -v"
    
    # 设置执行权限
    chmod +x bin/FacialExpressionAnalysis
else
    echo "❌ 构建失败: 找不到可执行文件"
    exit 1
fi

echo ""
echo "构建完成！"
