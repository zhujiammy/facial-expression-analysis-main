#include "facial_expression_dll.h"
#include "emotion_analyzer.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <cstring>

// 全局变量
static std::unique_ptr<EmotionAnalyzer> g_analyzer = nullptr;
static std::string g_last_error;

// 辅助函数：复制字符串到固定长度缓冲区
void safe_strcpy(char* dest, const char* src, size_t dest_size) {
    if (src == nullptr) {
        dest[0] = '\0';
        return;
    }
    strncpy_s(dest, dest_size, src, dest_size - 1);
    dest[dest_size - 1] = '\0';
}

// 辅助函数：设置错误信息
void set_error(const std::string& error) {
    g_last_error = error;
}

// 初始化情绪分析器
FACIAL_EXPRESSION_API int InitializeEmotionAnalyzer(
    const char* onnx_model_path,
    const char* shape_predictor_path,
    const char* frontalization_model_path
) {
    try {
        std::cout << "InitializeEmotionAnalyzer called with:" << std::endl;
        std::cout << "  ONNX: " << (onnx_model_path ? onnx_model_path : "NULL") << std::endl;
        std::cout << "  Shape: " << (shape_predictor_path ? shape_predictor_path : "NULL") << std::endl;
        std::cout << "  Front: " << (frontalization_model_path ? frontalization_model_path : "NULL") << std::endl;
        
        // 释放旧的实例
        if (g_analyzer) {
            g_analyzer.reset();
        }
        
        std::cout << "Creating EmotionAnalyzer instance..." << std::endl;
        g_analyzer = std::make_unique<EmotionAnalyzer>(
            onnx_model_path ? onnx_model_path : "model_emotion_pls30.onnx",
            frontalization_model_path ? frontalization_model_path : "model_frontalization.npy",
            shape_predictor_path ? shape_predictor_path : "shape_predictor_68_face_landmarks.dat"
        );
        
        std::cout << "EmotionAnalyzer instance created, calling initialize..." << std::endl;
        if (g_analyzer->initialize()) {
            set_error("");
            std::cout << "Initialization successful!" << std::endl;
            return 1; // 成功
        } else {
            set_error("Failed to initialize emotion analyzer");
            std::cout << "Initialization failed!" << std::endl;
            g_analyzer.reset();
            return 0; // 失败
        }
    } catch (const std::exception& e) {
        std::string error_msg = "Exception during initialization: " + std::string(e.what());
        set_error(error_msg);
        std::cout << error_msg << std::endl;
        g_analyzer.reset();
        return 0; // 失败
    } catch (...) {
        set_error("Unknown exception during initialization");
        std::cout << "Unknown exception during initialization" << std::endl;
        g_analyzer.reset();
        return 0; // 失败
    }
}

// 从文件分析情绪
FACIAL_EXPRESSION_API EmotionResultDLL AnalyzeEmotionFromFile(const char* image_path) {
    EmotionResultDLL result = { 0 };
    
    if (!g_analyzer) {
        safe_strcpy(result.error_message, "Emotion analyzer not initialized", sizeof(result.error_message));
        result.success = 0;
        return result;
    }
    
    if (!image_path) {
        safe_strcpy(result.error_message, "Image path is null", sizeof(result.error_message));
        result.success = 0;
        return result;
    }
    
    try {
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            safe_strcpy(result.error_message, "Failed to load image", sizeof(result.error_message));
            result.success = 0;
            return result;
        }
        
        EmotionResult emotion_result = g_analyzer->analyzeEmotion(image);
        
        result.arousal = emotion_result.arousal;
        result.valence = emotion_result.valence;
        result.intensity = emotion_result.intensity;
        safe_strcpy(result.emotion_name, emotion_result.emotion_name.c_str(), sizeof(result.emotion_name));
        result.success = 1;
        
        set_error("");
        
    } catch (const std::exception& e) {
        safe_strcpy(result.error_message, e.what(), sizeof(result.error_message));
        result.success = 0;
        set_error("Exception during emotion analysis: " + std::string(e.what()));
    }
    
    return result;
}

// 从字节数组分析情绪
FACIAL_EXPRESSION_API EmotionResultDLL AnalyzeEmotionFromBytes(
    const unsigned char* image_data,
    int data_length,
    int width,
    int height,
    int channels
) {
    EmotionResultDLL result = { 0 };
    
    if (!g_analyzer) {
        safe_strcpy(result.error_message, "Emotion analyzer not initialized", sizeof(result.error_message));
        result.success = 0;
        return result;
    }
    
    if (!image_data || data_length <= 0) {
        safe_strcpy(result.error_message, "Invalid image data", sizeof(result.error_message));
        result.success = 0;
        return result;
    }
    
    try {
        cv::Mat image;
        
        if (width > 0 && height > 0 && channels > 0) {
            // 从原始像素数据创建Mat
            int cv_type = (channels == 1) ? CV_8UC1 : (channels == 3) ? CV_8UC3 : CV_8UC4;
            image = cv::Mat(height, width, cv_type, (void*)image_data).clone();
        } else {
            // 从编码的图像数据（如JPEG, PNG）创建Mat
            std::vector<unsigned char> buffer(image_data, image_data + data_length);
            image = cv::imdecode(buffer, cv::IMREAD_COLOR);
        }
        
        if (image.empty()) {
            safe_strcpy(result.error_message, "Failed to decode image data", sizeof(result.error_message));
            result.success = 0;
            return result;
        }
        
        EmotionResult emotion_result = g_analyzer->analyzeEmotion(image);
        
        result.arousal = emotion_result.arousal;
        result.valence = emotion_result.valence;
        result.intensity = emotion_result.intensity;
        safe_strcpy(result.emotion_name, emotion_result.emotion_name.c_str(), sizeof(result.emotion_name));
        result.success = 1;
        
        set_error("");
        
    } catch (const std::exception& e) {
        safe_strcpy(result.error_message, e.what(), sizeof(result.error_message));
        result.success = 0;
        set_error("Exception during emotion analysis: " + std::string(e.what()));
    }
    
    return result;
}

// 释放资源
FACIAL_EXPRESSION_API void ReleaseEmotionAnalyzer() {
    g_analyzer.reset();
    set_error("");
}

// 获取最后的错误信息
FACIAL_EXPRESSION_API const char* GetLastError() {
    return g_last_error.c_str();
}

// 简单的测试函数实现
FACIAL_EXPRESSION_API int TestFunction() {
    return 42;
}

// 测试字符串参数的函数实现
FACIAL_EXPRESSION_API int TestStringFunction(const char* test_string) {
    if (test_string == nullptr) {
        return -1;
    }
    return strlen(test_string);
}
