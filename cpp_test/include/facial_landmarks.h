#pragma once

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <vector>

class FacialLandmarks {
public:
    // 将dlib landmarks转换为cv::Point2f向量
    static std::vector<cv::Point2f> dlibToOpenCV(const dlib::full_object_detection& landmarks);
    
    // 将cv::Point2f向量转换为dlib landmarks
    static dlib::full_object_detection openCVToDlib(const std::vector<cv::Point2f>& points, 
                                                   const dlib::rectangle& face_rect);
    
    // 计算landmarks的中心点
    static cv::Point2f calculateCenter(const std::vector<cv::Point2f>& landmarks);
    
    // 归一化landmarks到[-1,1]范围
    static std::vector<cv::Point2f> normalizeLandmarks(const std::vector<cv::Point2f>& landmarks);
    
    // 可视化landmarks
    static cv::Mat visualizeLandmarks(const cv::Mat& image, 
                                    const std::vector<cv::Point2f>& landmarks,
                                    const cv::Scalar& color = cv::Scalar(0, 255, 0));
    
    // 获取特定面部区域的landmarks索引
    static std::vector<int> getJawLineIndices();
    static std::vector<int> getRightEyebrowIndices();
    static std::vector<int> getLeftEyebrowIndices();
    static std::vector<int> getNoseIndices();
    static std::vector<int> getRightEyeIndices();
    static std::vector<int> getLeftEyeIndices();
    static std::vector<int> getMouthIndices();
    
    // 过滤掉下颚线landmarks（用于reduced feature模式）
    static std::vector<cv::Point2f> excludeJawLine(const std::vector<cv::Point2f>& landmarks);
    
private:
    // dlib 68个面部关键点的索引定义
    static const std::vector<int> JAW_LINE_INDICES;
    static const std::vector<int> RIGHT_EYEBROW_INDICES;
    static const std::vector<int> LEFT_EYEBROW_INDICES;
    static const std::vector<int> NOSE_INDICES;
    static const std::vector<int> RIGHT_EYE_INDICES;
    static const std::vector<int> LEFT_EYE_INDICES;
    static const std::vector<int> MOUTH_INDICES;
};
