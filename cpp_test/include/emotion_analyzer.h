#pragma once

#include <opencv2/opencv.hpp>

#ifdef DLIB_AVAILABLE
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#endif

#ifdef ONNX_AVAILABLE
#include <onnxruntime_cxx_api.h>
#endif

#ifdef CNPY_AVAILABLE
#include <cnpy.h>
#endif

#include <vector>
#include <string>
#include <memory>

struct EmotionResult {
    float arousal;
    float valence;
    float intensity;
    std::string emotion_name;
};

struct LandmarksData {
    std::vector<cv::Point2f> raw_landmarks;
    std::vector<cv::Point2f> frontal_landmarks;
};

class EmotionAnalyzer {
public:
    EmotionAnalyzer(const std::string& onnx_model_path,
                   const std::string& frontalization_model_path,
                   const std::string& shape_predictor_path);
    
    ~EmotionAnalyzer();
    
    // 初始化模型
    bool initialize();
    
    // 从图像中分析情感
    EmotionResult analyzeEmotion(const cv::Mat& image);
    
    // 获取面部关键点
    LandmarksData getFacialLandmarks(const cv::Mat& image);
    
    // 正面化关键点
    std::vector<cv::Point2f> frontalizeLandmarks(const std::vector<cv::Point2f>& landmarks);
    
    // 提取几何特征
    std::vector<float> extractGeometricFeatures(const std::vector<cv::Point2f>& landmarks);
    
    // 使用ONNX模型进行预测
    std::vector<float> predictWithONNX(const std::vector<float>& features);
    
    // 将AVI值转换为情感名称
    std::string aviToEmotionName(float arousal, float valence, float intensity = -1.0f);

private:
    // 私有成员变量
    std::string onnx_model_path_;
    std::string frontalization_model_path_;
    std::string shape_predictor_path_;
    
#ifdef ONNX_AVAILABLE
    // ONNX Runtime相关
    std::unique_ptr<Ort::Env> ort_env_;
    std::unique_ptr<Ort::Session> ort_session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    
    // ONNX 输入输出名称
    std::string input_name_;
    std::string output_name_;
#endif
    
    // 模型参数
    std::vector<float> frontalization_weights_; // Flattened 137x136 matrix
    bool full_features_;
    int components_;
    
#ifdef DLIB_AVAILABLE
    // dlib相关
    dlib::frontal_face_detector face_detector_;
    dlib::shape_predictor shape_predictor_;
#endif
    
    // 私有方法
    bool loadFrontalizationModel();
    bool loadONNXModel();
    bool loadShapePredictor();
    
    // 几何特征提取的辅助函数
    float calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2);
    float calculateScale(const std::vector<cv::Point2f>& landmarks, const std::vector<int>& landmark_indices);
    std::vector<cv::Point2f> procrustesStandardization(const std::vector<cv::Point2f>& landmarks);
    float calculateAngle(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3);
    std::vector<float> calculateDistanceFeatures(const std::vector<cv::Point2f>& landmarks);
    std::vector<float> calculateAngleFeatures(const std::vector<cv::Point2f>& landmarks);
    std::vector<float> calculateTriangleFeatures(const std::vector<cv::Point2f>& landmarks);
};
