#include "utils.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <map>

namespace Utils {

float calculateEuclideanDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

float calculateAngle(const cv::Point2f& p1, const cv::Point2f& center, const cv::Point2f& p2) {
    cv::Point2f v1 = p1 - center;
    cv::Point2f v2 = p2 - center;
    
    float dot_product = v1.x * v2.x + v1.y * v2.y;
    float mag1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
    float mag2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);
    
    if (mag1 == 0.0f || mag2 == 0.0f) {
        return 0.0f;
    }
    
    float cos_angle = dot_product / (mag1 * mag2);
    cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));
    
    return std::acos(cos_angle);
}

float calculateTriangleArea(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3) {
    return std::abs((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2.0f);
}

std::vector<float> normalizeVector(const std::vector<float>& input) {
    if (input.empty()) {
        return input;
    }
    
    float magnitude = vectorMagnitude(input);
    if (magnitude == 0.0f) {
        return input;
    }
    
    std::vector<float> normalized;
    normalized.reserve(input.size());
    
    for (float value : input) {
        normalized.push_back(value / magnitude);
    }
    
    return normalized;
}

float vectorMagnitude(const std::vector<float>& vec) {
    float sum = 0.0f;
    for (float value : vec) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

std::vector<float> vectorSubtract(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> result;
    if (a.size() != b.size()) {
        return result;
    }
    
    result.reserve(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result.push_back(a[i] - b[i]);
    }
    
    return result;
}

bool fileExists(const std::string& filepath) {
    std::ifstream file(filepath);
    return file.good();
}

std::string getFileExtension(const std::string& filepath) {
    size_t dot_pos = filepath.find_last_of('.');
    if (dot_pos != std::string::npos) {
        return filepath.substr(dot_pos + 1);
    }
    return "";
}

std::string getCurrentTimeString() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::vector<float> matToVector(const cv::Mat& mat) {
    std::vector<float> vec;
    
    if (mat.type() == CV_32F) {
        vec.assign(mat.ptr<float>(), mat.ptr<float>() + mat.total());
    } else if (mat.type() == CV_64F) {
        const double* data = mat.ptr<double>();
        vec.reserve(mat.total());
        for (size_t i = 0; i < mat.total(); ++i) {
            vec.push_back(static_cast<float>(data[i]));
        }
    }
    
    return vec;
}

cv::Mat vectorToMat(const std::vector<float>& vec, int rows, int cols) {
    if (vec.size() != static_cast<size_t>(rows * cols)) {
        return cv::Mat();
    }
    
    cv::Mat mat(rows, cols, CV_32F);
    std::memcpy(mat.data, vec.data(), vec.size() * sizeof(float));
    
    return mat;
}

std::vector<std::string> splitString(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}

std::string joinStrings(const std::vector<std::string>& strings, const std::string& delimiter) {
    if (strings.empty()) {
        return "";
    }
    
    std::stringstream ss;
    for (size_t i = 0; i < strings.size(); ++i) {
        ss << strings[i];
        if (i < strings.size() - 1) {
            ss << delimiter;
        }
    }
    
    return ss.str();
}

float calculateMean(const std::vector<float>& values) {
    if (values.empty()) {
        return 0.0f;
    }
    
    float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    return sum / values.size();
}

float calculateStdDev(const std::vector<float>& values) {
    if (values.empty()) {
        return 0.0f;
    }
    
    float mean = calculateMean(values);
    float variance = 0.0f;
    
    for (float value : values) {
        variance += (value - mean) * (value - mean);
    }
    
    variance /= values.size();
    return std::sqrt(variance);
}

float calculateMax(const std::vector<float>& values) {
    if (values.empty()) {
        return 0.0f;
    }
    
    return *std::max_element(values.begin(), values.end());
}

float calculateMin(const std::vector<float>& values) {
    if (values.empty()) {
        return 0.0f;
    }
    
    return *std::min_element(values.begin(), values.end());
}

std::string intensityToString(float intensity) {
    if (intensity < 0.1f) {
        return "neutral";
    } else if (intensity < 0.325f) {
        return "slightly";
    } else if (intensity < 0.55f) {
        return "moderately";
    } else if (intensity < 0.775f) {
        return "very";
    } else {
        return "extremely";
    }
}

cv::Scalar emotionToColor(const std::string& emotion) {
    static std::map<std::string, cv::Scalar> color_map = {
        {"happy", cv::Scalar(0, 255, 0)},      // 绿色
        {"sad", cv::Scalar(255, 0, 0)},        // 蓝色
        {"angry", cv::Scalar(0, 0, 255)},      // 红色
        {"surprised", cv::Scalar(0, 255, 255)}, // 黄色
        {"disgusted", cv::Scalar(128, 0, 128)}, // 紫色
        {"fearful", cv::Scalar(128, 128, 0)},   // 青色
        {"neutral", cv::Scalar(128, 128, 128)}  // 灰色
    };
    
    auto it = color_map.find(emotion);
    if (it != color_map.end()) {
        return it->second;
    }
    
    return cv::Scalar(255, 255, 255); // 默认白色
}

void printVector(const std::vector<float>& vec, const std::string& name) {
    if (!name.empty()) {
        std::cout << name << ": ";
    }
    
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

void printMatrix(const std::vector<std::vector<float>>& matrix, const std::string& name) {
    if (!name.empty()) {
        std::cout << name << ":" << std::endl;
    }
    
    for (size_t i = 0; i < matrix.size(); ++i) {
        std::cout << "  Row " << i << ": ";
        printVector(matrix[i]);
    }
}

std::map<std::string, std::string> readConfigFile(const std::string& filepath) {
    std::map<std::string, std::string> config;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        return config;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // 跳过空行和注释
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        size_t eq_pos = line.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = line.substr(0, eq_pos);
            std::string value = line.substr(eq_pos + 1);
            
            // 去除首尾空格
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            config[key] = value;
        }
    }
    
    file.close();
    return config;
}

} // namespace Utils
