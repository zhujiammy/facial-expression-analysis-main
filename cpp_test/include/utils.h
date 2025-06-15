#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace Utils {
    // 数学工具函数
    float calculateEuclideanDistance(const cv::Point2f& p1, const cv::Point2f& p2);
    float calculateAngle(const cv::Point2f& p1, const cv::Point2f& center, const cv::Point2f& p2);
    float calculateTriangleArea(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3);
    
    // 向量操作
    std::vector<float> normalizeVector(const std::vector<float>& input);
    float vectorMagnitude(const std::vector<float>& vec);
    std::vector<float> vectorSubtract(const std::vector<float>& a, const std::vector<float>& b);
    
    // 文件IO
    bool fileExists(const std::string& filepath);
    std::string getFileExtension(const std::string& filepath);
    std::string getCurrentTimeString();
    
    // 数据转换
    std::vector<float> matToVector(const cv::Mat& mat);
    cv::Mat vectorToMat(const std::vector<float>& vec, int rows, int cols);
    
    // 字符串工具
    std::vector<std::string> splitString(const std::string& str, char delimiter);
    std::string joinStrings(const std::vector<std::string>& strings, const std::string& delimiter);
    
    // 统计函数
    float calculateMean(const std::vector<float>& values);
    float calculateStdDev(const std::vector<float>& values);
    float calculateMax(const std::vector<float>& values);
    float calculateMin(const std::vector<float>& values);
    
    // 情感分析相关
    std::string intensityToString(float intensity);
    cv::Scalar emotionToColor(const std::string& emotion);
    
    // 调试和日志
    void printVector(const std::vector<float>& vec, const std::string& name = "");
    void printMatrix(const std::vector<std::vector<float>>& matrix, const std::string& name = "");
    
    // 配置文件读取（简单的key=value格式）
    std::map<std::string, std::string> readConfigFile(const std::string& filepath);
}
