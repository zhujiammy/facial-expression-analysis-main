#include <iostream>
#include <windows.h>

// 直接定义结构体，避免包含头文件
struct EmotionResultDLL {
    float arousal;
    float valence;
    float intensity;
    char emotion_name[128];
    int success;
    char error_message[256];
};

// 函数指针类型定义
typedef int (*InitializeEmotionAnalyzerFunc)(const char*, const char*, const char*);
typedef EmotionResultDLL (*AnalyzeEmotionFromFileFunc)(const char*);
typedef void (*ReleaseEmotionAnalyzerFunc)();
typedef const char* (*GetLastErrorFunc)();

int main() {
    std::cout << "=== DLL 直接测试程序 ===" << std::endl;
    
    // 加载DLL
    HMODULE hDll = LoadLibraryA("FacialExpressionDLL.dll");
    if (!hDll) {
        std::cerr << "无法加载DLL，错误代码: " << GetLastError() << std::endl;
        return 1;
    }
    
    std::cout << "DLL加载成功" << std::endl;
    
    // 获取函数指针
    InitializeEmotionAnalyzerFunc initFunc = 
        (InitializeEmotionAnalyzerFunc)GetProcAddress(hDll, "InitializeEmotionAnalyzer");
    AnalyzeEmotionFromFileFunc analyzeFunc = 
        (AnalyzeEmotionFromFileFunc)GetProcAddress(hDll, "AnalyzeEmotionFromFile");
    ReleaseEmotionAnalyzerFunc releaseFunc = 
        (ReleaseEmotionAnalyzerFunc)GetProcAddress(hDll, "ReleaseEmotionAnalyzer");
    GetLastErrorFunc getErrorFunc = 
        (GetLastErrorFunc)GetProcAddress(hDll, "GetLastError");
    
    if (!initFunc) {
        std::cerr << "无法获取InitializeEmotionAnalyzer函数" << std::endl;
        FreeLibrary(hDll);
        return 1;
    }
    
    if (!analyzeFunc) {
        std::cerr << "无法获取AnalyzeEmotionFromFile函数" << std::endl;
        FreeLibrary(hDll);
        return 1;
    }
    
    if (!releaseFunc) {
        std::cerr << "无法获取ReleaseEmotionAnalyzer函数" << std::endl;
        FreeLibrary(hDll);
        return 1;
    }
    
    if (!getErrorFunc) {
        std::cerr << "无法获取GetLastError函数" << std::endl;
        FreeLibrary(hDll);
        return 1;
    }
    
    std::cout << "所有函数获取成功" << std::endl;
      // 测试初始化
    std::cout << "正在初始化情绪分析器..." << std::endl;
    int initResult = initFunc(
        "model_emotion_pls30.onnx",              // ONNX模型路径
        "shape_predictor_68_face_landmarks.dat", // dlib关键点检测器路径
        "model_frontalization.npy"               // 正面化模型路径
    );
    
    std::cout << "初始化结果: " << initResult << std::endl;
    
    if (initResult == 0) {
        const char* error = getErrorFunc();
        std::cout << "初始化失败，错误信息: " << (error ? error : "无错误信息") << std::endl;
    } else {
        std::cout << "初始化成功！" << std::endl;        // 测试分析图片
        std::cout << "正在分析图片..." << std::endl;
        EmotionResultDLL result = analyzeFunc("../../../data/images/pleased.jpg");
        
        if (result.success == 1) {
            std::cout << "分析成功！" << std::endl;
            std::cout << "  Arousal: " << result.arousal << std::endl;
            std::cout << "  Valence: " << result.valence << std::endl;
            std::cout << "  Intensity: " << result.intensity << std::endl;
            std::cout << "  情绪: " << result.emotion_name << std::endl;
        } else {
            std::cout << "分析失败，错误信息: " << result.error_message << std::endl;
        }
    }
    
    // 释放资源
    std::cout << "正在释放资源..." << std::endl;
    releaseFunc();
    
    FreeLibrary(hDll);
    std::cout << "测试完成" << std::endl;
    
    return 0;
}
