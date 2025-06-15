#include "emotion_analyzer.h"
#include "facial_landmarks.h"
#include "utils.h"
#include <iostream>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef CNPY_AVAILABLE
#include <cnpy.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

EmotionAnalyzer::EmotionAnalyzer(const std::string& onnx_model_path,
                               const std::string& frontalization_model_path,
                               const std::string& shape_predictor_path)
    : onnx_model_path_(onnx_model_path)
    , frontalization_model_path_(frontalization_model_path)    , shape_predictor_path_(shape_predictor_path)
    , full_features_(false)
    , components_(30)
#ifdef DLIB_AVAILABLE
    , face_detector_(dlib::get_frontal_face_detector())
#endif
{
}

EmotionAnalyzer::~EmotionAnalyzer() = default;

bool EmotionAnalyzer::initialize() {
    std::cout << "Initializing emotion analyzer..." << std::endl;
    
    try {
#ifdef ONNX_AVAILABLE
        // Initialize ONNX Runtime environment with compatibility handling
        std::cout << "Initializing ONNX Runtime environment..." << std::endl;
        try {
            // Use a more conservative environment initialization
            ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "EmotionAnalyzer");
            std::cout << "ONNX Runtime environment created successfully" << std::endl;
            
            session_options_ = std::make_unique<Ort::SessionOptions>();
            std::cout << "ONNX Runtime session options created successfully" << std::endl;
            
            // Set conservative settings for better compatibility
            session_options_->SetIntraOpNumThreads(1);
            session_options_->SetInterOpNumThreads(1);
            
            // Disable graph optimization for better compatibility
            session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
            
            // Disable CPU execution provider extensions that might cause issues
            session_options_->DisableCpuMemArena();
            session_options_->DisableMemPattern();
            
            std::cout << "ONNX Runtime environment initialized successfully" << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime exception: " << e.what() << std::endl;
            return false;
        } catch (const std::exception& e) {
            std::cerr << "ONNX Runtime initialization failed: " << e.what() << std::endl;
            return false;
        } catch (...) {
            std::cerr << "ONNX Runtime initialization failed: Unknown exception" << std::endl;
            return false;
        }
        
        // Load models
        std::cout << "Loading ONNX model..." << std::endl;
        if (!loadONNXModel()) {
            std::cerr << "Failed to load ONNX model" << std::endl;
            return false;
        }
        std::cout << "ONNX model loaded successfully" << std::endl;
#else
        std::cerr << "ONNX Runtime not available" << std::endl;
        return false;
#endif
        
        if (!loadFrontalizationModel()) {
            std::cerr << "Failed to load frontalization model" << std::endl;
            return false;
        }
        
#ifdef DLIB_AVAILABLE
        if (!loadShapePredictor()) {
            std::cerr << "Failed to load shape predictor" << std::endl;
            return false;
        }
#else
        std::cerr << "Warning: dlib not available - using simple face detection fallback" << std::endl;
#endif
        
        std::cout << "Emotion analyzer initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

bool EmotionAnalyzer::loadONNXModel() {
#ifdef ONNX_AVAILABLE
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
          // Get input and output names
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_ptr = ort_session_->GetInputNameAllocated(0, allocator);
        auto output_name_ptr = ort_session_->GetOutputNameAllocated(0, allocator);
        
        input_name_ = std::string(input_name_ptr.get());
        output_name_ = std::string(output_name_ptr.get());
        
        std::cout << "ONNX model loaded successfully" << std::endl;
        std::cout << "Input name: " << input_name_ << std::endl;
        std::cout << "Output name: " << output_name_ << std::endl;
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
#else
    std::cerr << "ONNX Runtime not available" << std::endl;
    return false;
#endif
}

bool EmotionAnalyzer::loadFrontalizationModel() {
    // Load frontalization model from .npy file
    std::cout << "Loading frontalization model from " << frontalization_model_path_ << std::endl;
    
#ifdef CNPY_AVAILABLE
    try {
        cnpy::NpyArray arr = cnpy::npy_load(frontalization_model_path_);
        
        std::cout << "Loaded array with word_size: " << arr.word_size << std::endl;
        
        // Expected shape: (137, 136) for DLIB 68 landmarks
        // 137 = 2*68 + 1 (for intercept), 136 = 2*68
        if (arr.shape.size() != 2 || arr.shape[0] != 137 || arr.shape[1] != 136) {
            std::cerr << "Invalid frontalization model shape. Expected (137, 136), got (" 
                     << arr.shape[0] << ", " << arr.shape[1] << ")" << std::endl;
            return false;
        }
        
        // Handle both float32 and float64
        if (arr.word_size == sizeof(double)) {
            // Convert from float64 to float32
            double* data = arr.data<double>();
            frontalization_weights_.reserve(arr.num_vals);
            for (size_t i = 0; i < arr.num_vals; i++) {
                frontalization_weights_.push_back(static_cast<float>(data[i]));
            }
        } else if (arr.word_size == sizeof(float)) {
            // Use float32 directly
            float* data = arr.data<float>();
            frontalization_weights_.assign(data, data + arr.num_vals);
        } else {
            std::cerr << "Unsupported data type in frontalization model" << std::endl;
            return false;
        }
        
        std::cout << "Frontalization model loaded successfully" << std::endl;
        std::cout << "Model shape: (" << arr.shape[0] << ", " << arr.shape[1] << ")" << std::endl;
        std::cout << "Sample weights: " << frontalization_weights_[0] << ", " 
                  << frontalization_weights_[1] << ", " << frontalization_weights_[2] << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load frontalization model: " << e.what() << std::endl;
        return false;
    }
#else
    std::cout << "CNPY not available, skipping frontalization model loading" << std::endl;
    // Create a dummy identity-like transformation for fallback
    frontalization_weights_.resize(137 * 136, 0.0f);
    
    // Set up identity mapping (simplified fallback)
    for (int i = 0; i < 136; i++) {
        frontalization_weights_[i * 137 + i] = 1.0f;
    }
    
    return true;
#endif
}

bool EmotionAnalyzer::loadShapePredictor() {
#ifdef DLIB_AVAILABLE
    try {
        dlib::deserialize(shape_predictor_path_) >> shape_predictor_;
        std::cout << "Shape predictor loaded successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load shape predictor: " << e.what() << std::endl;
        return false;
    }
#else
    std::cerr << "dlib not available for shape predictor" << std::endl;
    return false;
#endif
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
        
        if (prediction.size() >= 2) {
            result.arousal = prediction[0];
            result.valence = prediction[1];
            
            // Apply limits to arousal and valence (same as Python)
            if (result.arousal > 1.0f) result.arousal = 1.0f;
            else if (result.arousal < -1.0f) result.arousal = -1.0f;
            
            if (result.valence > 1.0f) result.valence = 1.0f;
            else if (result.valence < -1.0f) result.valence = -1.0f;
            
            // Calculate intensity as Euclidean distance (same as Python)
            result.intensity = std::sqrt(result.valence * result.valence + result.arousal * result.arousal);
            if (result.intensity > 1.0f) result.intensity = 1.0f;
            else if (result.intensity < 0.0f) result.intensity = 0.0f;
            
            // Round to 3 decimal places (same as Python)
            result.intensity = std::round(result.intensity * 1000.0f) / 1000.0f;
            
            result.emotion_name = aviToEmotionName(result.arousal, result.valence, result.intensity);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in emotion analysis: " << e.what() << std::endl;
    }
    
    return result;
}

LandmarksData EmotionAnalyzer::getFacialLandmarks(const cv::Mat& image) {
    LandmarksData result;
    
#ifdef DLIB_AVAILABLE
    try {
        dlib::cv_image<dlib::bgr_pixel> dlib_image(image);
        std::vector<dlib::rectangle> faces = face_detector_(dlib_image);
        
        if (!faces.empty()) {
            dlib::full_object_detection landmarks = shape_predictor_(dlib_image, faces[0]);
            
            std::cout << "Detected " << landmarks.num_parts() << " landmarks:" << std::endl;
            
            for (int i = 0; i < landmarks.num_parts(); ++i) {
                dlib::point p = landmarks.part(i);
                result.raw_landmarks.push_back(cv::Point2f(p.x(), p.y()));
                
                // Print first few landmarks for debugging
                if (i < 10) {
                    std::cout << "  Landmark " << i << ": (" << p.x() << ", " << p.y() << ")" << std::endl;
                }
            }
        } else {
            std::cout << "No faces detected" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error detecting facial landmarks: " << e.what() << std::endl;
    }
#else
    std::cerr << "dlib not available - cannot detect facial landmarks" << std::endl;
    // Fallback: create dummy landmarks for center of image
    int height = image.rows;
    int width = image.cols;
    float cx = width / 2.0f;
    float cy = height / 2.0f;
    
    // Create 68 dummy landmarks arranged in a rough face shape
    for (int i = 0; i < 68; ++i) {
        result.raw_landmarks.push_back(cv::Point2f(cx, cy));
    }
#endif
    
    return result;
}

std::vector<cv::Point2f> EmotionAnalyzer::frontalizeLandmarks(const std::vector<cv::Point2f>& landmarks) {
    // Implement frontalization using the loaded model
    if (landmarks.size() != 68 || frontalization_weights_.empty()) {
        std::cout << "Frontalization not available, using original landmarks" << std::endl;
        return landmarks; // Return original landmarks if no model
    }
      // Step 1: Apply Procrustes standardization
    std::vector<cv::Point2f> standardized = procrustesStandardization(landmarks);
    
    // Debug: print first few standardized landmarks
    std::cout << "First few standardized landmarks: ";
    for (size_t i = 0; i < std::min(standardized.size(), size_t(5)); ++i) {
        std::cout << "(" << standardized[i].x << ", " << standardized[i].y << ")";
        if (i < 4) std::cout << ", ";
    }
    std::cout << std::endl;
    
    // Step 2: Create feature vector with intercept
    // Format: [x1, x2, ..., x68, y1, y2, ..., y68, 1]
    std::vector<float> feature_vector;
    feature_vector.reserve(137); // 68*2 + 1
    
    // Add all X coordinates
    for (const auto& point : standardized) {
        feature_vector.push_back(point.x);
    }
    
    // Add all Y coordinates  
    for (const auto& point : standardized) {
        feature_vector.push_back(point.y);
    }
    
    // Add intercept term
    feature_vector.push_back(1.0f);
      // Step 3: Apply frontalization transformation
    // Python: np.matmul(feature_vector, frontalization_weights)
    // feature_vector: (137,) weights: (137, 136) -> output: (136,)
    // result[i] = sum(feature_vector[j] * weights[j][i] for j in range(137))
    std::vector<float> frontal_vector(136, 0.0f);
    
    for (int i = 0; i < 136; i++) {
        for (int j = 0; j < 137; j++) {
            // frontalization_weights_ is stored in row-major order
            // weights[j][i] = frontalization_weights_[j * 136 + i]
            frontal_vector[i] += feature_vector[j] * frontalization_weights_[j * 136 + i];
        }
    }
    
    // Step 4: Convert back to landmark points
    std::vector<cv::Point2f> frontal_landmarks;
    frontal_landmarks.reserve(68);
    
    for (int i = 0; i < 68; i++) {
        float x = frontal_vector[i];      // X coordinates: 0-67
        float y = frontal_vector[i + 68]; // Y coordinates: 68-135
        frontal_landmarks.push_back(cv::Point2f(x, y));
    }
      std::cout << "Applied frontalization to landmarks" << std::endl;
    
    // Debug: print first few frontal landmarks
    std::cout << "First few frontal landmarks: ";
    for (size_t i = 0; i < std::min(frontal_landmarks.size(), size_t(5)); ++i) {
        std::cout << "(" << frontal_landmarks[i].x << ", " << frontal_landmarks[i].y << ")";
        if (i < 4) std::cout << ", ";
    }
    std::cout << std::endl;
    
    return frontal_landmarks;
}

std::vector<float> EmotionAnalyzer::extractGeometricFeatures(const std::vector<cv::Point2f>& landmarks) {
    std::vector<float> features;
    
    if (landmarks.size() < 68) {
        return features;
    }
    
    // IMPORTANT: Match the Python bug exactly!
    // For full_features=False, Python uses:
    // - Feature extraction: landmarks 0-50 (first 51 landmarks) - BUG!
    // - Scale calculation: landmarks 17-67 (correct)
    
    std::vector<int> feature_landmark_indices;
    std::vector<int> scale_landmark_indices;
    
    if (full_features_) {
        // Use all 68 landmarks (0-67) for both
        for (int i = 0; i < 68; i++) {
            feature_landmark_indices.push_back(i);
            scale_landmark_indices.push_back(i);
        }
        std::cout << "Using all 68 landmarks (0-67)" << std::endl;
    } else {
        // MATCH PYTHON BUG: Use landmarks 0-50 for features, 17-67 for scale
        for (int i = 0; i < 51; i++) {  // 51 landmarks (0-50)
            feature_landmark_indices.push_back(i);
        }
        for (int i = 17; i < 68; i++) {  // 51 landmarks (17-67)
            scale_landmark_indices.push_back(i);
        }
        std::cout << "Using landmarks 0-50 for features, 17-67 for scale (matching Python bug), feature_count=" << feature_landmark_indices.size() << std::endl;
    }
    
    // Calculate scale for normalization using the scale landmarks
    float scale = calculateScale(landmarks, scale_landmark_indices);
    
    // Calculate normalized distances between all pairs of FEATURE landmarks (N choose 2)
    for (size_t i = 0; i < feature_landmark_indices.size(); i++) {
        for (size_t j = i + 1; j < feature_landmark_indices.size(); j++) {
            int idx1 = feature_landmark_indices[i];
            int idx2 = feature_landmark_indices[j];
            
            if (idx1 < static_cast<int>(landmarks.size()) && idx2 < static_cast<int>(landmarks.size())) {
                float distance = calculateDistance(landmarks[idx1], landmarks[idx2]);
                // Normalize by scale (same as Python implementation)
                distance /= scale;
                features.push_back(distance);
                
                // Debug: print first few feature pairs with more detail
                if (features.size() <= 5) {
                    cv::Point2f p1 = landmarks[idx1];
                    cv::Point2f p2 = landmarks[idx2];
                    float raw_distance = calculateDistance(p1, p2);
                    std::cout << "Feature " << features.size() << ": landmarks[" << idx1 << "] (" << p1.x << "," << p1.y 
                              << ") and landmarks[" << idx2 << "] (" << p2.x << "," << p2.y 
                              << ") -> raw_dist=" << raw_distance << ", normalized=" << distance << std::endl;
                }
            }
        }
    }
      std::cout << "Extracted " << features.size() << " normalized geometric features (scale=" << scale << ")" << std::endl;
    
    // Debug: print first 10 features for comparison with Python
    std::cout << "First 10 features: ";
    for (size_t i = 0; i < std::min(features.size(), size_t(10)); ++i) {
        std::cout << features[i];
        if (i < 9) std::cout << ", ";
    }
    std::cout << std::endl;
    
    // Debug: print min/max feature values
    if (!features.empty()) {
        auto minmax = std::minmax_element(features.begin(), features.end());
        std::cout << "Feature min/max: " << *minmax.first << " / " << *minmax.second << std::endl;
    }
    
    return features;
}

float EmotionAnalyzer::calculateScale(const std::vector<cv::Point2f>& landmarks, const std::vector<int>& landmark_indices) {
    // Compute scale as mean euclidean distance of all landmarks to the mean landmark
    // This matches the Python get_scale function
    
    if (landmark_indices.empty()) {
        return 1.0f;
    }
    
    // Calculate mean landmark position
    float mean_x = 0.0f, mean_y = 0.0f;
    for (int idx : landmark_indices) {
        if (idx < static_cast<int>(landmarks.size())) {
            mean_x += landmarks[idx].x;
            mean_y += landmarks[idx].y;
        }
    }
    mean_x /= landmark_indices.size();
    mean_y /= landmark_indices.size();
    
    // Calculate mean squared distance to mean point
    float sum_squared_distances = 0.0f;
    for (int idx : landmark_indices) {
        if (idx < static_cast<int>(landmarks.size())) {
            float dx = landmarks[idx].x - mean_x;
            float dy = landmarks[idx].y - mean_y;
            sum_squared_distances += (dx * dx + dy * dy);
        }
    }
    
    float mean_squared_distance = sum_squared_distances / landmark_indices.size();
    return std::sqrt(mean_squared_distance);
}

std::vector<float> EmotionAnalyzer::predictWithONNX(const std::vector<float>& features) {
    std::vector<float> result;
    
#ifdef ONNX_AVAILABLE
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
        std::vector<const char*> input_names = {input_name_.c_str()};
        std::vector<const char*> output_names = {output_name_.c_str()};
        
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
#else
    std::cerr << "ONNX Runtime not available - cannot make predictions" << std::endl;
    // Return dummy values
    result = {0.0f, 0.0f}; // arousal, valence
#endif
    
    return result;
}

std::string EmotionAnalyzer::aviToEmotionName(float arousal, float valence, float intensity) {
    // Exact implementation of Python's avi_to_text function
    // Based on Russell's Circumplex Model of Affect
    
    std::vector<std::string> ls_expr_intensity = {
        "Slightly", "Moderately", "Very", "Extremely"
    };
    
    std::vector<std::string> ls_expr_name = {
        "pleased", "happy", "delighted", "excited", "astonished", 
        "aroused", // first quarter
        
        "tensed", "alarmed", "afraid", "annoyed", "distressed", 
        "frustrated", "miserable", // second quarter
        
        "sad", "gloomy", "depressed", "bored", "droopy", "tired", 
        "sleepy", // third quarter
        
        "calm", "serene", "content", "satisfied"  // fourth quarter
    };
    
    std::string expression_intensity = "";
    std::string expression_name = "";
    
    // If intensity not provided, calculate it
    if (intensity < 0) {
        intensity = std::sqrt(arousal * arousal + valence * valence);
    }
    
    // Analyzing intensity
    if (intensity < 0.1f) {
        expression_name = "Neutral";
        expression_intensity = "";
    } else {
        // Determine intensity level
        if (intensity < 0.325f) {
            expression_intensity = ls_expr_intensity[0];
        } else if (intensity < 0.55f) {
            expression_intensity = ls_expr_intensity[1];
        } else if (intensity < 0.775f) {
            expression_intensity = ls_expr_intensity[2];
        } else {
            expression_intensity = ls_expr_intensity[3];
        }
        
        // Analyzing expression name - compute angle [0,360]
        float theta;
        if (valence == 0.0f) {
            if (arousal >= 0.0f) {
                theta = 90.0f;
            } else {
                theta = 270.0f;
            }
        } else {
            theta = std::atan(arousal / valence);
            theta = theta * (180.0f / M_PI);
            
            if (valence < 0.0f) {
                theta = 180.0f + theta;
            } else if (arousal < 0.0f) {
                theta = 360.0f + theta;
            }
        }
        
        // Estimate expression name based on theta ranges (same as Python)
        if (theta < 16.0f || theta > 354.0f) {
            expression_name = ls_expr_name[0];  // pleased
        } else if (theta < 34.0f) {
            expression_name = ls_expr_name[1];  // happy
        } else if (theta < 62.5f) {
            expression_name = ls_expr_name[2];  // delighted
        } else if (theta < 78.5f) {
            expression_name = ls_expr_name[3];  // excited
        } else if (theta < 93.0f) {
            expression_name = ls_expr_name[4];  // astonished
        } else if (theta < 104.0f) {
            expression_name = ls_expr_name[5];  // aroused
        } else if (theta < 115.0f) {
            expression_name = ls_expr_name[6];  // tensed
        } else if (theta < 126.0f) {
            expression_name = ls_expr_name[7];  // alarmed
        } else if (theta < 137.0f) {
            expression_name = ls_expr_name[8];  // afraid
        } else if (theta < 148.0f) {
            expression_name = ls_expr_name[9];  // annoyed
        } else if (theta < 159.0f) {
            expression_name = ls_expr_name[10]; // distressed
        } else if (theta < 170.0f) {
            expression_name = ls_expr_name[11]; // frustrated
        } else if (theta < 181.0f) {
            expression_name = ls_expr_name[12]; // miserable
        } else if (theta < 192.0f) {
            expression_name = ls_expr_name[13]; // sad
        } else if (theta < 203.0f) {
            expression_name = ls_expr_name[14]; // gloomy
        } else if (theta < 215.0f) {
            expression_name = ls_expr_name[15]; // depressed
        } else if (theta < 230.0f) {
            expression_name = ls_expr_name[16]; // bored
        } else if (theta < 245.0f) {
            expression_name = ls_expr_name[17]; // droopy
        } else if (theta < 260.0f) {
            expression_name = ls_expr_name[18]; // tired
        } else if (theta < 280.0f) {
            expression_name = ls_expr_name[19]; // sleepy
        } else if (theta < 300.0f) {
            expression_name = ls_expr_name[20]; // calm
        } else if (theta < 320.0f) {
            expression_name = ls_expr_name[21]; // serene
        } else if (theta < 340.0f) {
            expression_name = ls_expr_name[22]; // content
        } else if (theta < 354.0f) {
            expression_name = ls_expr_name[23]; // satisfied
        } else {
            expression_name = "Unknown";
            expression_intensity = "";
        }
    }
    
    // Return combined expression (same format as Python)
    if (expression_intensity.empty()) {
        return expression_name;
    } else {
        return expression_intensity + " " + expression_name;
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

std::vector<cv::Point2f> EmotionAnalyzer::procrustesStandardization(const std::vector<cv::Point2f>& landmarks) {
    // Implement Procrustes analysis for landmark standardization
    // This matches the get_procrustes function in Python
    
    if (landmarks.size() != 68) {
        return landmarks; // Return original if not 68 landmarks
    }
    
    std::vector<cv::Point2f> landmarks_standard = landmarks;
      // Step 1: Translation - center the landmarks
    cv::Point2f centroid(0, 0);
    for (const auto& point : landmarks) {
        centroid.x += point.x;
        centroid.y += point.y;
    }
    centroid.x /= landmarks.size();
    centroid.y /= landmarks.size();
    
    std::cout << "Mean landmark: (" << centroid.x << ", " << centroid.y << ")" << std::endl;
    
    for (auto& point : landmarks_standard) {
        point.x -= centroid.x;
        point.y -= centroid.y;
    }
    
    std::cout << "First few centered landmarks: ";
    for (size_t i = 0; i < std::min(landmarks_standard.size(), size_t(5)); ++i) {
        std::cout << "(" << landmarks_standard[i].x << ", " << landmarks_standard[i].y << ")";
        if (i < 4) std::cout << ", ";
    }
    std::cout << std::endl;    // Step 2: Scale - normalize by mean distance from origin
    // Python: landmark_scale = sqrt(mean(sum(landmarks_standard**2, axis=1)))
    float sum_squared_distances = 0.0f;
    for (const auto& point : landmarks_standard) {
        sum_squared_distances += (point.x * point.x + point.y * point.y);
    }
    float scale = std::sqrt(sum_squared_distances / landmarks.size());
    
    std::cout << "Calculated scale: " << scale << std::endl;
    
    if (scale > 0) {
        for (auto& point : landmarks_standard) {
            point.x /= scale;
            point.y /= scale;
        }
    }
    
    std::cout << "First few scaled landmarks: ";
    for (size_t i = 0; i < std::min(landmarks_standard.size(), size_t(5)); ++i) {
        std::cout << "(" << landmarks_standard[i].x << ", " << landmarks_standard[i].y << ")";
        if (i < 4) std::cout << ", ";
    }
    std::cout << std::endl;
      // Step 3: Rotation - rotate to align eyes horizontally
    // Calculate eye centers (matching Python's get_eye_centers)
    cv::Point2f center_eye_left(0, 0), center_eye_right(0, 0);
    
    // Left eye landmarks: 36-41
    for (int i = 36; i < 42; i++) {
        center_eye_left.x += landmarks_standard[i].x;
        center_eye_left.y += landmarks_standard[i].y;
    }
    center_eye_left.x /= 6;
    center_eye_left.y /= 6;
    
    // Right eye landmarks: 42-47
    for (int i = 42; i < 48; i++) {
        center_eye_right.x += landmarks_standard[i].x;
        center_eye_right.y += landmarks_standard[i].y;
    }
    center_eye_right.x /= 6;
    center_eye_right.y /= 6;
    
    std::cout << "Eye centers: left(" << center_eye_left.x << ", " << center_eye_left.y 
              << "), right(" << center_eye_right.x << ", " << center_eye_right.y << ")" << std::endl;
    
    // Calculate rotation angle
    float dx = center_eye_right.x - center_eye_left.x;
    float dy = center_eye_right.y - center_eye_left.y;
    
    std::cout << "Eye distance: dx=" << dx << ", dy=" << dy << std::endl;
    
    if (dx != 0) {
        float angle = std::atan(dy / dx);
        std::cout << "Rotation angle: " << angle << " radians (" << (angle * 180.0f / M_PI) << " degrees)" << std::endl;
          // Create rotation matrix and apply to landmarks
        // Python: R = [[cos(a), -sin(a)], [sin(a), cos(a)]]
        // landmarks_new = landmarks @ R
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);
        
        // Apply rotation to all landmarks
        for (auto& point : landmarks_standard) {
            float x_new = point.x * cos_a + point.y * sin_a;
            float y_new = -point.x * sin_a + point.y * cos_a;
            point.x = x_new;
            point.y = y_new;
        }
    }
    
    return landmarks_standard;
}
