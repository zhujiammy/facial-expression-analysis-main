@echo off
echo === 运行面部表情分析 C# 测试程序 ===

cd /d "d:\pythonpro\facial-expression-analysis-main\csharp_test"

:: 确保所有必要的文件都在正确位置
echo 检查文件...

if not exist "bin\FacialExpressionDLL.dll" (
    echo 错误: 找不到 FacialExpressionDLL.dll
    echo 请先运行 build_all.bat 构建项目
    pause
    exit /b 1
)

if not exist "bin\model_emotion_pls30.onnx" (
    echo 错误: 找不到模型文件 model_emotion_pls30.onnx
    echo 请确保模型文件已复制到 bin 目录
    pause
    exit /b 1
)

echo 所有文件检查完毕，正在运行测试程序...
echo.

:: 切换到bin目录运行，确保DLL和模型文件在同一目录
cd bin
..\bin\Release\net6.0\FacialExpressionTest.exe

pause
