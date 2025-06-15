#pragma once

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#include <vector>
#include <string>
#include <memory>

#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

class EmotionAnalyzer {
public:
    EmotionAnalyzer(const std::string& onnx_model_path,
                   const std::string& frontalization_model_path,
                   const std::string& shape_predictor_path);
    
    ~EmotionAnalyzer();
    
    bool initialize();
    
    std::vector<float> analyzeEmotion(const cv::Mat& image);
    
    struct EmotionResult {
        std::vector<float> probabilities;
        std::string predicted_emotion;
        float confidence;
        bool success;
        std::string error_message;
    };
    
    EmotionResult predictEmotion(const cv::Mat& image);
    
    bool setFullFeatures(bool full_features);
    bool setComponents(int components);
    
    std::vector<std::string> getEmotionLabels() const;

private:
    std::string onnx_model_path_;
    std::string frontalization_model_path_;
    std::string shape_predictor_path_;
    
    bool full_features_;
    int components_;
    
    dlib::frontal_face_detector detector_;
    dlib::shape_predictor sp_;
    
    std::vector<std::vector<float>> frontalization_model_;
    
#ifdef USE_ONNXRUNTIME
    std::unique_ptr<Ort::Env> ort_env_;
    std::unique_ptr<Ort::Session> ort_session_;
    std::unique_ptr<Ort::SessionOptions> ort_session_options_;
#endif
    
    std::vector<std::string> emotion_labels_;
    
    bool loadONNXModel();
    bool loadFrontalizationModel();
    bool loadShapePredictor();
    
    std::vector<float> extractFeatures(const cv::Mat& image);
    std::vector<float> detectFacialLandmarks(const cv::Mat& image);
    std::vector<float> frontalizeFace(const std::vector<float>& landmarks);
    std::vector<float> applyPCA(const std::vector<float>& features);
    
    cv::Mat preprocessImage(const cv::Mat& image);
    
    std::vector<cv::Point2f> dlibToOpenCV(const dlib::full_object_detection& landmarks);
    
    double calculateAngle(const cv::Point2f& p1, const cv::Point2f& p2);
    
    std::vector<float> calculateGeometricFeatures(const std::vector<cv::Point2f>& landmarks);
    std::vector<float> calculateDistanceFeatures(const std::vector<cv::Point2f>& landmarks);
    std::vector<float> calculateRatioFeatures(const std::vector<cv::Point2f>& landmarks);
    std::vector<float> calculateAngleFeatures(const std::vector<cv::Point2f>& landmarks);
    
    void initializeEmotionLabels();
};
