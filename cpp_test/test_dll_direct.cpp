#include <iostream>
#include "facial_expression_dll.h"

int main() {
    std::cout << "测试DLL接口..." << std::endl;
    
    // 测试初始化
    std::cout << "正在初始化..." << std::endl;
    int result = InitializeEmotionAnalyzer(
        "model_emotion_pls30.onnx",
        "shape_predictor_68_face_landmarks.dat", 
        "model_frontalization.npy"
    );
    
    std::cout << "初始化结果: " << result << std::endl;
    
    if (result == 0) {
        const char* error = GetLastError();
        if (error && strlen(error) > 0) {
            std::cout << "错误信息: " << error << std::endl;
        }
    } else {
        std::cout << "初始化成功！" << std::endl;
        
        // 测试文件分析
        std::cout << "\n测试图片分析..." << std::endl;
        EmotionResultDLL emotion = AnalyzeEmotionFromFile("../data/images/pleased.jpg");
        
        if (emotion.success) {
            std::cout << "分析成功!" << std::endl;
            std::cout << "Arousal: " << emotion.arousal << std::endl;
            std::cout << "Valence: " << emotion.valence << std::endl;
            std::cout << "Intensity: " << emotion.intensity << std::endl;
            std::cout << "Emotion: " << emotion.emotion_name << std::endl;
        } else {
            std::cout << "分析失败: " << emotion.error_message << std::endl;
        }
    }
    
    // 释放资源
    ReleaseEmotionAnalyzer();
    std::cout << "测试完成" << std::endl;
    
    return 0;
}
