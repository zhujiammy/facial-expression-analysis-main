#pragma once

#ifdef FACIAL_EXPRESSION_EXPORTS
#define FACIAL_EXPRESSION_API __declspec(dllexport)
#else
#define FACIAL_EXPRESSION_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// 测试字符串参数的函数
FACIAL_EXPRESSION_API int __cdecl TestStringFunction(const char* test_string);

// 简单的测试函数
FACIAL_EXPRESSION_API int __cdecl TestFunction();

// 情绪分析结果结构体
typedef struct {
    float arousal;
    float valence;
    float intensity;
    char emotion_name[128];
    int success;
    char error_message[256];
} EmotionResultDLL;

// DLL接口函数声明
FACIAL_EXPRESSION_API int __cdecl InitializeEmotionAnalyzer(
    const char* onnx_model_path,
    const char* shape_predictor_path,
    const char* frontalization_model_path
);

FACIAL_EXPRESSION_API EmotionResultDLL __cdecl AnalyzeEmotionFromFile(const char* image_path);

FACIAL_EXPRESSION_API EmotionResultDLL __cdecl AnalyzeEmotionFromBytes(
    const unsigned char* image_data,
    int data_length,
    int width,
    int height,
    int channels
);

FACIAL_EXPRESSION_API void __cdecl ReleaseEmotionAnalyzer();

FACIAL_EXPRESSION_API const char* __cdecl GetLastError();

#ifdef __cplusplus
}
#endif
