@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   面部表情分析 - C++ 项目构建脚本
echo ========================================

:: 设置变量
set BUILD_DIR=build
set CMAKE_GENERATOR="Visual Studio 16 2019"
set VCPKG_TOOLCHAIN=C:\vcpkg\scripts\buildsystems\vcpkg.cmake

:: 检查是否在正确的目录
if not exist "CMakeLists.txt" (
    echo 错误: 请在包含 CMakeLists.txt 的目录中运行此脚本
    pause
    exit /b 1
)

:: 创建构建目录
if not exist "%BUILD_DIR%" (
    echo 创建构建目录: %BUILD_DIR%
    mkdir "%BUILD_DIR%"
)

cd "%BUILD_DIR%"

:: 检查 vcpkg 工具链
if exist "%VCPKG_TOOLCHAIN%" (
    echo 使用 vcpkg 工具链: %VCPKG_TOOLCHAIN%
    set CMAKE_ARGS=-DCMAKE_TOOLCHAIN_FILE=%VCPKG_TOOLCHAIN%
) else (
    echo 警告: 未找到 vcpkg 工具链，使用系统默认设置
    set CMAKE_ARGS=
)

:: 运行 CMake 配置
echo.
echo 配置项目...
cmake .. %CMAKE_ARGS% -A x64
if !ERRORLEVEL! neq 0 (
    echo 错误: CMake 配置失败
    pause
    exit /b 1
)

:: 编译项目
echo.
echo 编译项目...
cmake --build . --config Release
if !ERRORLEVEL! neq 0 (
    echo 错误: 编译失败
    pause
    exit /b 1
)

:: 检查构建结果
if exist "bin\Release\FacialExpressionAnalysis.exe" (
    echo.
    echo ✅ 构建成功！
    echo 可执行文件位置: %cd%\bin\Release\FacialExpressionAnalysis.exe
    echo.
    echo 使用示例:
    echo   %cd%\bin\Release\FacialExpressionAnalysis.exe -h
    echo   %cd%\bin\Release\FacialExpressionAnalysis.exe -i ..\..\data\images\pleased.jpg
    echo   %cd%\bin\Release\FacialExpressionAnalysis.exe -v
) else (
    echo ❌ 构建失败: 找不到可执行文件
    pause
    exit /b 1
)

echo.
echo 构建完成！
pause
