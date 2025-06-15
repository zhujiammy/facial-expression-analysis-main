#include "emotion_analyzer.h"
#include "facial_landmarks.h"
#include "utils.h"
#include <iostream>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

EmotionAnalyzer::EmotionAnalyzer(const std::string& onnx_model_path,
                               const std::string& frontalization_model_path,
                               const std::string& shape_predictor_path)
    : onnx_model_path_(onnx_model_path)
    , frontalization_model_path_(frontalization_model_path)
    , shape_predictor_path_(shape_predictor_path)
    , full_features_(false)
    , components_(30) {
}

EmotionAnalyzer::~EmotionAnalyzer() = default;

bool EmotionAnalyzer::initialize() {
    std::cout << "Initializing emotion analyzer..." << std::endl;
    
    try {
        // Initialize ONNX Runtime environment
        ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "EmotionAnalyzer");
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(1);
        
        // Load models
        if (!loadONNXModel()) {
            std::cerr << "Failed to load ONNX model" << std::endl;
            return false;
        }
        
        if (!loadFrontalizationModel()) {
            std::cerr << "Failed to load frontalization model" << std::endl;
            return false;
        }
        
        if (!loadShapePredictor()) {
            std::cerr << "Failed to load shape predictor" << std::endl;
            return false;
        }
        
        std::cout << "Emotion analyzer initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

bool EmotionAnalyzer::loadONNXModel() {
    try {
        // Load ONNX model
        #ifdef _WIN32
        std::wstring wide_path(onnx_model_path_.begin(), onnx_model_path_.end());
        ort_session_ = std::make_unique<Ort::Session>(*ort_env_, wide_path.c_str(), *session_options_);
        #else
        ort_session_ = std::make_unique<Ort::Session>(*ort_env_, onnx_model_path_.c_str(), *session_options_);
        #endif
        
        // Get input/output info
        auto input_info = ort_session_->GetInputTypeInfo(0);
        auto tensor_info = input_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        
        std::cout << "ONNX model loaded successfully" << std::endl;
        std::cout << "Input dimensions: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load ONNX model: " << e.what() << std::endl;
        return false;
    }
}

bool EmotionAnalyzer::loadFrontalizationModel() {
    // Load frontalization model from .npy file
    std::cout << "Loading frontalization model from " << frontalization_model_path_ << std::endl;
    
    // For now, just return true - implement actual loading later
    return true;
}

bool EmotionAnalyzer::loadShapePredictor() {
    try {
        dlib::deserialize(shape_predictor_path_) >> shape_predictor_;
        std::cout << "Shape predictor loaded successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load shape predictor: " << e.what() << std::endl;
        return false;
    }
}

EmotionResult EmotionAnalyzer::analyzeEmotion(const cv::Mat& image) {
    EmotionResult result;
    result.arousal = 0.0f;
    result.valence = 0.0f;
    result.intensity = 0.0f;
    result.emotion_name = "neutral";
    
    try {
        // Get facial landmarks
        auto landmarks_data = getFacialLandmarks(image);
        
        if (landmarks_data.raw_landmarks.empty()) {
            std::cerr << "No face detected in image" << std::endl;
            return result;
        }
        
        // Frontalize landmarks
        auto frontal_landmarks = frontalizeLandmarks(landmarks_data.raw_landmarks);
        
        // Extract geometric features
        auto features = extractGeometricFeatures(frontal_landmarks);
        
        if (features.empty()) {
            std::cerr << "Failed to extract features" << std::endl;
            return result;
        }
        
        // Predict with ONNX model
        auto prediction = predictWithONNX(features);
        
        if (prediction.size() >= 3) {
            result.arousal = prediction[0];
            result.valence = prediction[1];
            result.intensity = prediction.size() > 2 ? prediction[2] : 0.0f;
            result.emotion_name = aviToEmotionName(result.arousal, result.valence, result.intensity);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in emotion analysis: " << e.what() << std::endl;
    }
    
    return result;
}

LandmarksData EmotionAnalyzer::getFacialLandmarks(const cv::Mat& image) {
    LandmarksData result;
    
    try {
        dlib::cv_image<dlib::bgr_pixel> dlib_image(image);
        std::vector<dlib::rectangle> faces = face_detector_(dlib_image);
        
        if (!faces.empty()) {
            dlib::full_object_detection landmarks = shape_predictor_(dlib_image, faces[0]);
            
            for (int i = 0; i < landmarks.num_parts(); ++i) {
                dlib::point p = landmarks.part(i);
                result.raw_landmarks.push_back(cv::Point2f(p.x(), p.y()));
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error detecting facial landmarks: " << e.what() << std::endl;
    }
    
    return result;
}

std::vector<cv::Point2f> EmotionAnalyzer::frontalizeLandmarks(const std::vector<cv::Point2f>& landmarks) {
    // Simplified frontalization - just return the original landmarks for now
    return landmarks;
}

std::vector<float> EmotionAnalyzer::extractGeometricFeatures(const std::vector<cv::Point2f>& landmarks) {
    std::vector<float> features;
    
    if (landmarks.size() < 68) {
        return features;
    }
    
    // Calculate basic geometric features
    // Distance features
    auto dist_features = calculateDistanceFeatures(landmarks);
    features.insert(features.end(), dist_features.begin(), dist_features.end());
    
    // Angle features
    auto angle_features = calculateAngleFeatures(landmarks);
    features.insert(features.end(), angle_features.begin(), angle_features.end());
    
    // Triangle features
    auto triangle_features = calculateTriangleFeatures(landmarks);
    features.insert(features.end(), triangle_features.begin(), triangle_features.end());
    
    return features;
}

std::vector<float> EmotionAnalyzer::predictWithONNX(const std::vector<float>& features) {
    std::vector<float> result;
    
    try {
        // Create input tensor
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(features.size())};
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            const_cast<float*>(features.data()), 
            features.size(),
            input_shape.data(), 
            input_shape.size()
        );
        
        // Run inference
        std::vector<const char*> input_names = {"input"};
        std::vector<const char*> output_names = {"output"};
        
        auto output_tensors = ort_session_->Run(
            Ort::RunOptions{nullptr}, 
            input_names.data(),
            &input_tensor, 
            1, 
            output_names.data(),
            1
        );
        
        // Extract results
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        int output_size = 1;
        for (auto dim : output_shape) {
            output_size *= dim;
        }
        
        result.assign(output_data, output_data + output_size);
        
    } catch (const std::exception& e) {
        std::cerr << "ONNX prediction failed: " << e.what() << std::endl;
    }
    
    return result;
}

std::string EmotionAnalyzer::aviToEmotionName(float arousal, float valence, float intensity) {
    // Simple emotion mapping based on arousal and valence
    if (valence > 0.5f) {
        if (arousal > 0.5f) return "excited";
        else return "happy";
    } else if (valence < -0.5f) {
        if (arousal > 0.5f) return "angry";
        else return "sad";
    } else {
        if (arousal > 0.7f) return "surprised";
        else return "neutral";
    }
}

float EmotionAnalyzer::calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

float EmotionAnalyzer::calculateAngle(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3) {
    cv::Point2f v1 = p1 - p2;
    cv::Point2f v2 = p3 - p2;
    
    float dot = v1.x * v2.x + v1.y * v2.y;
    float cross = v1.x * v2.y - v1.y * v2.x;
    
    return std::atan2(cross, dot) * 180.0f / M_PI;
}

std::vector<float> EmotionAnalyzer::calculateDistanceFeatures(const std::vector<cv::Point2f>& landmarks) {
    std::vector<float> features;
    
    // Example: distances between key points
    if (landmarks.size() >= 68) {
        // Eye distances
        features.push_back(calculateDistance(landmarks[36], landmarks[39])); // Left eye width
        features.push_back(calculateDistance(landmarks[42], landmarks[45])); // Right eye width
        
        // Mouth distances
        features.push_back(calculateDistance(landmarks[48], landmarks[54])); // Mouth width
        features.push_back(calculateDistance(landmarks[51], landmarks[57])); // Mouth height
        
        // Nose distances
        features.push_back(calculateDistance(landmarks[31], landmarks[35])); // Nose width
    }
    
    return features;
}

std::vector<float> EmotionAnalyzer::calculateAngleFeatures(const std::vector<cv::Point2f>& landmarks) {
    std::vector<float> features;
    
    if (landmarks.size() >= 68) {
        // Eyebrow angles
        features.push_back(calculateAngle(landmarks[17], landmarks[19], landmarks[21])); // Left eyebrow
        features.push_back(calculateAngle(landmarks[22], landmarks[24], landmarks[26])); // Right eyebrow
        
        // Mouth angles
        features.push_back(calculateAngle(landmarks[48], landmarks[51], landmarks[54])); // Mouth curve
    }
    
    return features;
}

std::vector<float> EmotionAnalyzer::calculateTriangleFeatures(const std::vector<cv::Point2f>& landmarks) {
    std::vector<float> features;
    
    if (landmarks.size() >= 68) {
        // Triangle areas
        auto area = [](const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3) {
            return 0.5f * std::abs((p1.x - p3.x) * (p2.y - p1.y) - (p1.x - p2.x) * (p3.y - p1.y));
        };
        
        // Eye triangles
        features.push_back(area(landmarks[36], landmarks[37], landmarks[41])); // Left eye
        features.push_back(area(landmarks[42], landmarks[43], landmarks[47])); // Right eye
        
        // Mouth triangle
        features.push_back(area(landmarks[48], landmarks[51], landmarks[54])); // Mouth
    }
    
    return features;
}
