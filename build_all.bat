@echo off
echo === 构建面部表情分析 DLL 和 C# 测试程序 ===

cd /d "d:\pythonpro\facial-expression-analysis-main\cpp_test"

echo 正在构建 C++ DLL...
if exist build_dll (
    rmdir /s /q build_dll
)
mkdir build_dll
cd build_dll

cmake -G "Visual Studio 17 2022" -A x64 ..
if %ERRORLEVEL% NEQ 0 (
    echo CMAKE 配置失败！
    pause
    exit /b 1
)

cmake --build . --config Release
if %ERRORLEVEL% NEQ 0 (
    echo DLL 构建失败！
    pause
    exit /b 1
)

echo DLL 构建成功！

echo.
echo 正在复制文件到 C# 测试目录...
cd /d "d:\pythonpro\facial-expression-analysis-main"

:: 创建C#测试程序的bin目录
if not exist "csharp_test\bin" mkdir "csharp_test\bin"

:: 复制DLL和依赖文件
copy "cpp_test\build_dll\bin\FacialExpressionDLL.dll" "csharp_test\bin\" >nul 2>&1
copy "cpp_test\build_dll\bin\*.dll" "csharp_test\bin\" >nul 2>&1

:: 复制模型文件
copy "models\*.onnx" "csharp_test\bin\" >nul 2>&1
copy "models\*.npy" "csharp_test\bin\" >nul 2>&1
copy "models\*.dat" "csharp_test\bin\" >nul 2>&1

echo.
echo 正在构建 C# 测试程序...
cd csharp_test

dotnet build -c Release
if %ERRORLEVEL% NEQ 0 (
    echo C# 程序构建失败！
    pause
    exit /b 1
)

echo.
echo 构建完成！
echo.
echo DLL 位置: cpp_test\build_dll\bin\FacialExpressionDLL.dll
echo C# 测试程序: csharp_test\bin\Release\net6.0\FacialExpressionTest.exe
echo.
echo 运行测试请执行: run_test.bat
echo.
pause
