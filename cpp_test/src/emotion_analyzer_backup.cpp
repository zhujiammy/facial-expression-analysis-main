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
        // 初始化ONNX Runtime环境
        ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "EmotionAnalyzer");
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(1);
        
        // 加载各个模型
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
        // 加载ONNX模型
        #ifdef _WIN32
        std::wstring wide_path(onnx_model_path_.begin(), onnx_model_path_.end());
        ort_session_ = std::make_unique<Ort::Session>(*ort_env_, wide_path.c_str(), *session_options_);
        #else
        ort_session_ = std::make_unique<Ort::Session>(*ort_env_, onnx_model_path_.c_str(), *session_options_);
        #endif
        
        // 获取输入输出信息
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
        std::cerr << "ONNX模型加载失败: " << e.what() << std::endl;
        return false;
    }
}

bool EmotionAnalyzer::loadFrontalizationModel() {
    try {
        // 使用cnpy加载.npy文件
        cnpy::NpyArray arr = cnpy::npy_load(frontalization_model_path_);
        
        if (arr.shape.size() != 2) {
            std::cerr << "正面化模型维度错误，期望2D数组" << std::endl;
            return false;
        }
        
        int rows = arr.shape[0];
        int cols = arr.shape[1];
        
        std::cout << "正面化模型形状: [" << rows << ", " << cols << "]" << std::endl;
        
        // 转换数据
        float* data = arr.data<float>();
        
        // 初始化权重矩阵和偏置向量
        frontalization_weights_.resize(rows - 1);
        for (int i = 0; i < rows - 1; ++i) {
            frontalization_weights_[i].resize(cols);
            for (int j = 0; j < cols; ++j) {
                frontalization_weights_[i][j] = data[i * cols + j];
            }
        }
        
        // 最后一行是偏置
        frontalization_bias_.resize(cols);
        for (int j = 0; j < cols; ++j) {
            frontalization_bias_[j] = data[(rows - 1) * cols + j];
        }
        
        std::cout << "正面化模型加载成功" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "正面化模型加载失败: " << e.what() << std::endl;
        return false;
    }
}

bool EmotionAnalyzer::loadShapePredictor() {
    try {
        // 初始化面部检测器
        face_detector_ = dlib::get_frontal_face_detector();
        
        // 加载68点面部关键点检测器
        dlib::deserialize(shape_predictor_path_) >> shape_predictor_;
        
        std::cout << "面部关键点检测器加载成功" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "面部关键点检测器加载失败: " << e.what() << std::endl;
        return false;
    }
}

EmotionResult EmotionAnalyzer::analyzeEmotion(const cv::Mat& image) {
    EmotionResult result = {0.0f, 0.0f, 0.0f, "neutral"};
    
    try {
        // 获取面部关键点
        LandmarksData landmarks_data = getFacialLandmarks(image);
        
        if (landmarks_data.raw_landmarks.empty()) {
            std::cerr << "未检测到面部关键点" << std::endl;
            return result;
        }
        
        // 正面化关键点
        std::vector<cv::Point2f> frontal_landmarks = frontalizeLandmarks(landmarks_data.raw_landmarks);
        
        // 提取几何特征
        std::vector<float> features = extractGeometricFeatures(frontal_landmarks);
        
        // 使用ONNX模型预测
        std::vector<float> predictions = predictWithONNX(features);
        
        if (predictions.size() >= 2) {
            result.arousal = std::max(-1.0f, std::min(1.0f, predictions[0]));
            result.valence = std::max(-1.0f, std::min(1.0f, predictions[1]));
            result.intensity = std::sqrt(result.arousal * result.arousal + result.valence * result.valence);
            result.intensity = std::max(0.0f, std::min(1.0f, result.intensity));
            result.emotion_name = aviToEmotionName(result.arousal, result.valence, result.intensity);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "情感分析失败: " << e.what() << std::endl;
    }
    
    return result;
}

LandmarksData EmotionAnalyzer::getFacialLandmarks(const cv::Mat& image) {
    LandmarksData result;
    
    try {
        // 转换为dlib图像格式
        dlib::cv_image<dlib::bgr_pixel> dlib_image(image);
        
        // 检测面部
        std::vector<dlib::rectangle> faces = face_detector_(dlib_image);
        
        if (faces.empty()) {
            std::cerr << "未检测到面部" << std::endl;
            return result;
        }
        
        // 使用第一个检测到的面部
        dlib::full_object_detection landmarks = shape_predictor_(dlib_image, faces[0]);
        
        // 转换为OpenCV格式
        result.raw_landmarks = FacialLandmarks::dlibToOpenCV(landmarks);
        
    } catch (const std::exception& e) {
        std::cerr << "面部关键点检测失败: " << e.what() << std::endl;
    }
    
    return result;
}

std::vector<cv::Point2f> EmotionAnalyzer::frontalizeLandmarks(const std::vector<cv::Point2f>& landmarks) {
    std::vector<cv::Point2f> frontal_landmarks;
    
    try {
        if (landmarks.size() != 68) {
            std::cerr << "关键点数量错误，期望68个点" << std::endl;
            return frontal_landmarks;
        }
        
        // 将landmarks转换为特征向量 (x1, y1, x2, y2, ...)
        std::vector<float> input_features;
        for (const auto& point : landmarks) {
            input_features.push_back(point.x);
            input_features.push_back(point.y);
        }
        
        // 应用正面化变换 y = Wx + b
        std::vector<float> output_features(frontalization_bias_);
        
        for (size_t i = 0; i < frontalization_weights_.size(); ++i) {
            for (size_t j = 0; j < frontalization_weights_[i].size(); ++j) {
                output_features[j] += frontalization_weights_[i][j] * input_features[i];
            }
        }
        
        // 转换回关键点格式
        frontal_landmarks.clear();
        for (size_t i = 0; i < output_features.size(); i += 2) {
            frontal_landmarks.emplace_back(output_features[i], output_features[i + 1]);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "关键点正面化失败: " << e.what() << std::endl;
    }
    
    return frontal_landmarks;
}

std::vector<float> EmotionAnalyzer::extractGeometricFeatures(const std::vector<cv::Point2f>& landmarks) {
    std::vector<float> features;
    
    try {
        // 根据配置决定是否使用完整特征集
        std::vector<cv::Point2f> working_landmarks = landmarks;
        
        if (!full_features_) {
            // 排除下颚线（前17个点）
            working_landmarks = std::vector<cv::Point2f>(landmarks.begin() + 17, landmarks.end());
        }
        
        // 提取距离特征
        std::vector<float> distance_features = calculateDistanceFeatures(working_landmarks);
        features.insert(features.end(), distance_features.begin(), distance_features.end());
        
        // 提取角度特征
        std::vector<float> angle_features = calculateAngleFeatures(working_landmarks);
        features.insert(features.end(), angle_features.begin(), angle_features.end());
        
        // 提取三角形特征
        std::vector<float> triangle_features = calculateTriangleFeatures(working_landmarks);
        features.insert(features.end(), triangle_features.begin(), triangle_features.end());
        
        std::cout << "提取特征维度: " << features.size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "特征提取失败: " << e.what() << std::endl;
    }
    
    return features;
}

std::vector<float> EmotionAnalyzer::predictWithONNX(const std::vector<float>& features) {
    std::vector<float> predictions;
    
    try {
        // 准备输入数据
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(features.size())};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            const_cast<float*>(features.data()), 
            features.size(), 
            input_shape.data(), 
            input_shape.size()
        );
          // 运行推理
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
        
        // 提取结果
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        int output_size = 1;
        for (auto dim : output_shape) {
            output_size *= dim;
        }
        
        predictions.assign(output_data, output_data + output_size);
        
    } catch (const std::exception& e) {
        std::cerr << "ONNX预测失败: " << e.what() << std::endl;
    }
    
    return predictions;
}

// 实现距离特征计算
std::vector<float> EmotionAnalyzer::calculateDistanceFeatures(const std::vector<cv::Point2f>& landmarks) {
    std::vector<float> distances;
    
    // 计算所有关键点对之间的欧氏距离
    for (size_t i = 0; i < landmarks.size(); ++i) {
        for (size_t j = i + 1; j < landmarks.size(); ++j) {
            float dist = calculateDistance(landmarks[i], landmarks[j]);
            distances.push_back(dist);
        }
    }
    
    return distances;
}

std::vector<float> EmotionAnalyzer::calculateAngleFeatures(const std::vector<cv::Point2f>& landmarks) {
    std::vector<float> angles;
    
    // 计算三点角度特征
    for (size_t i = 0; i < landmarks.size(); ++i) {
        for (size_t j = 0; j < landmarks.size(); ++j) {
            for (size_t k = 0; k < landmarks.size(); ++k) {
                if (i != j && j != k && i != k) {
                    float angle = calculateAngle(landmarks[i], landmarks[j], landmarks[k]);
                    angles.push_back(angle);
                }
            }
        }
    }
    
    return angles;
}

std::vector<float> EmotionAnalyzer::calculateTriangleFeatures(const std::vector<cv::Point2f>& landmarks) {
    std::vector<float> triangles;
    
    // 计算三角形面积特征
    for (size_t i = 0; i < landmarks.size(); ++i) {
        for (size_t j = i + 1; j < landmarks.size(); ++j) {
            for (size_t k = j + 1; k < landmarks.size(); ++k) {
                // 计算三角形面积
                float area = std::abs(
                    (landmarks[i].x * (landmarks[j].y - landmarks[k].y) +
                     landmarks[j].x * (landmarks[k].y - landmarks[i].y) +
                     landmarks[k].x * (landmarks[i].y - landmarks[j].y)) / 2.0f
                );
                triangles.push_back(area);
            }
        }
    }
    
    return triangles;
}

float EmotionAnalyzer::calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

float EmotionAnalyzer::calculateAngle(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3) {
    cv::Point2f v1 = p1 - p2;
    cv::Point2f v2 = p3 - p2;
    
    float dot_product = v1.x * v2.x + v1.y * v2.y;
    float mag1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
    float mag2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);
    
    if (mag1 == 0.0f || mag2 == 0.0f) return 0.0f;
    
    float cos_angle = dot_product / (mag1 * mag2);
    cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));
    
    return std::acos(cos_angle);
}

std::string EmotionAnalyzer::aviToEmotionName(float arousal, float valence, float intensity) {
    if (intensity < 0.0f) {
        intensity = std::sqrt(arousal * arousal + valence * valence);
    }
    
    if (intensity < 0.1f) {
        return "neutral";
    }
    
    // 根据Russell's Circumplex Model转换为情感标签
    float theta;
    if (valence == 0) {
        theta = (arousal >= 0) ? 90 : 270;
    } else {
        theta = std::atan(arousal / valence) * 180.0f / M_PI;
        if (valence < 0) theta += 180;
        else if (arousal < 0) theta += 360;
    }
    
    // 强度描述
    std::string intensity_desc;
    if (intensity < 0.325f) intensity_desc = "slightly";
    else if (intensity < 0.55f) intensity_desc = "moderately";
    else if (intensity < 0.775f) intensity_desc = "very";
    else intensity_desc = "extremely";
    
    // 情感名称
    std::string emotion_name;
    if (theta < 16 || theta > 354) emotion_name = "pleased";
    else if (theta < 34) emotion_name = "happy";
    else if (theta < 52) emotion_name = "delighted";
    else if (theta < 70) emotion_name = "excited";
    else if (theta < 88) emotion_name = "astonished";
    else if (theta < 106) emotion_name = "aroused";
    else if (theta < 124) emotion_name = "tensed";
    else if (theta < 142) emotion_name = "alarmed";
    else if (theta < 160) emotion_name = "afraid";
    else if (theta < 178) emotion_name = "annoyed";
    else if (theta < 196) emotion_name = "distressed";
    else if (theta < 214) emotion_name = "frustrated";
    else if (theta < 232) emotion_name = "miserable";
    else if (theta < 250) emotion_name = "sad";
    else if (theta < 268) emotion_name = "gloomy";
    else if (theta < 286) emotion_name = "depressed";
    else if (theta < 304) emotion_name = "bored";
    else if (theta < 322) emotion_name = "droopy";
    else if (theta < 340) emotion_name = "tired";
    else emotion_name = "sleepy";
    
    return intensity_desc + " " + emotion_name;
}
