#include <windows.h>
#include <iostream>

// 加载DLL并测试基本功能
typedef int (*InitFunc)(const char*, const char*, const char*);
typedef const char* (*GetErrorFunc)();
typedef void (*ReleaseFunc)();

int main() {
    std::cout << "正在加载DLL..." << std::endl;
    
    HMODULE hDll = LoadLibrary(L"FacialExpressionDLL.dll");
    if (!hDll) {
        std::cerr << "无法加载DLL, 错误代码: " << GetLastError() << std::endl;
        return 1;
    }
    
    std::cout << "DLL加载成功" << std::endl;
    
    // 获取函数指针
    InitFunc initFunc = (InitFunc)GetProcAddress(hDll, "InitializeEmotionAnalyzer");
    GetErrorFunc getErrorFunc = (GetErrorFunc)GetProcAddress(hDll, "GetLastError");
    ReleaseFunc releaseFunc = (ReleaseFunc)GetProcAddress(hDll, "ReleaseEmotionAnalyzer");
    
    if (!initFunc || !getErrorFunc || !releaseFunc) {
        std::cerr << "无法获取DLL函数" << std::endl;
        FreeLibrary(hDll);
        return 1;
    }
    
    std::cout << "正在初始化..." << std::endl;
    
    // 测试初始化
    int result = initFunc("model_emotion_pls30.onnx", "shape_predictor_68_face_landmarks.dat", "model_frontalization.npy");
    
    std::cout << "初始化结果: " << result << std::endl;
    
    if (result == 0) {
        const char* error = getErrorFunc();
        if (error) {
            std::cout << "错误信息: " << error << std::endl;
        }
    }
    
    // 释放资源
    releaseFunc();
    
    FreeLibrary(hDll);
    std::cout << "测试完成" << std::endl;
    
    return 0;
}
