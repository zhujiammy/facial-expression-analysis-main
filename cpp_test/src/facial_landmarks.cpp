#include "facial_landmarks.h"
#include <algorithm>

// 静态成员初始化
const std::vector<int> FacialLandmarks::JAW_LINE_INDICES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
const std::vector<int> FacialLandmarks::RIGHT_EYEBROW_INDICES = {17, 18, 19, 20, 21};
const std::vector<int> FacialLandmarks::LEFT_EYEBROW_INDICES = {22, 23, 24, 25, 26};
const std::vector<int> FacialLandmarks::NOSE_INDICES = {27, 28, 29, 30, 31, 32, 33, 34, 35};
const std::vector<int> FacialLandmarks::RIGHT_EYE_INDICES = {36, 37, 38, 39, 40, 41};
const std::vector<int> FacialLandmarks::LEFT_EYE_INDICES = {42, 43, 44, 45, 46, 47};
const std::vector<int> FacialLandmarks::MOUTH_INDICES = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67};

std::vector<cv::Point2f> FacialLandmarks::dlibToOpenCV(const dlib::full_object_detection& landmarks) {
    std::vector<cv::Point2f> points;
    points.reserve(landmarks.num_parts());
    
    for (int i = 0; i < landmarks.num_parts(); ++i) {
        dlib::point p = landmarks.part(i);
        points.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()));
    }
    
    return points;
}

dlib::full_object_detection FacialLandmarks::openCVToDlib(const std::vector<cv::Point2f>& points, 
                                                        const dlib::rectangle& face_rect) {
    // 创建一个包含所有关键点的 dlib::full_object_detection
    std::vector<dlib::point> dlib_points;
    for (const auto& point : points) {
        dlib_points.push_back(dlib::point(static_cast<long>(point.x), static_cast<long>(point.y)));
    }
    
    return dlib::full_object_detection(face_rect, dlib_points);
}

cv::Point2f FacialLandmarks::calculateCenter(const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.empty()) {
        return cv::Point2f(0, 0);
    }
    
    float sum_x = 0, sum_y = 0;
    for (const auto& point : landmarks) {
        sum_x += point.x;
        sum_y += point.y;
    }
    
    return cv::Point2f(sum_x / landmarks.size(), sum_y / landmarks.size());
}

std::vector<cv::Point2f> FacialLandmarks::normalizeLandmarks(const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.empty()) {
        return landmarks;
    }
    
    // 计算边界框
    float min_x = landmarks[0].x, max_x = landmarks[0].x;
    float min_y = landmarks[0].y, max_y = landmarks[0].y;
    
    for (const auto& point : landmarks) {
        min_x = std::min(min_x, point.x);
        max_x = std::max(max_x, point.x);
        min_y = std::min(min_y, point.y);
        max_y = std::max(max_y, point.y);
    }
    
    float width = max_x - min_x;
    float height = max_y - min_y;
    float scale = std::max(width, height);
    
    if (scale == 0) {
        return landmarks;
    }
    
    // 归一化到[-1, 1]范围
    cv::Point2f center((min_x + max_x) / 2, (min_y + max_y) / 2);
    std::vector<cv::Point2f> normalized;
    normalized.reserve(landmarks.size());
    
    for (const auto& point : landmarks) {
        float norm_x = (point.x - center.x) / (scale / 2);
        float norm_y = (point.y - center.y) / (scale / 2);
        normalized.emplace_back(norm_x, norm_y);
    }
    
    return normalized;
}

cv::Mat FacialLandmarks::visualizeLandmarks(const cv::Mat& image, 
                                          const std::vector<cv::Point2f>& landmarks,
                                          const cv::Scalar& color) {
    cv::Mat result = image.clone();
    
    // 绘制关键点
    for (size_t i = 0; i < landmarks.size(); ++i) {
        cv::circle(result, landmarks[i], 2, color, -1);
        // 可选：添加点的索引标签
        cv::putText(result, std::to_string(i), 
                   cv::Point(landmarks[i].x + 3, landmarks[i].y - 3),
                   cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1);
    }
    
    // 连接面部轮廓线
    auto drawContour = [&](const std::vector<int>& indices, bool closed = false) {
        for (size_t i = 0; i < indices.size() - 1; ++i) {
            if (indices[i] < landmarks.size() && indices[i + 1] < landmarks.size()) {
                cv::line(result, landmarks[indices[i]], landmarks[indices[i + 1]], color, 1);
            }
        }
        if (closed && !indices.empty() && indices[0] < landmarks.size() && indices.back() < landmarks.size()) {
            cv::line(result, landmarks[indices.back()], landmarks[indices[0]], color, 1);
        }
    };
    
    // 绘制面部轮廓
    drawContour(JAW_LINE_INDICES);
    drawContour(RIGHT_EYEBROW_INDICES);
    drawContour(LEFT_EYEBROW_INDICES);
    drawContour(NOSE_INDICES);
    drawContour(RIGHT_EYE_INDICES, true);
    drawContour(LEFT_EYE_INDICES, true);
    
    // 嘴部需要分内外轮廓
    std::vector<int> outer_mouth = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
    std::vector<int> inner_mouth = {60, 61, 62, 63, 64, 65, 66, 67};
    drawContour(outer_mouth, true);
    drawContour(inner_mouth, true);
    
    return result;
}

std::vector<int> FacialLandmarks::getJawLineIndices() {
    return JAW_LINE_INDICES;
}

std::vector<int> FacialLandmarks::getRightEyebrowIndices() {
    return RIGHT_EYEBROW_INDICES;
}

std::vector<int> FacialLandmarks::getLeftEyebrowIndices() {
    return LEFT_EYEBROW_INDICES;
}

std::vector<int> FacialLandmarks::getNoseIndices() {
    return NOSE_INDICES;
}

std::vector<int> FacialLandmarks::getRightEyeIndices() {
    return RIGHT_EYE_INDICES;
}

std::vector<int> FacialLandmarks::getLeftEyeIndices() {
    return LEFT_EYE_INDICES;
}

std::vector<int> FacialLandmarks::getMouthIndices() {
    return MOUTH_INDICES;
}

std::vector<cv::Point2f> FacialLandmarks::excludeJawLine(const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.size() < 68) {
        return landmarks;
    }
    
    // 返回除了前17个点(下颚线)之外的所有点
    return std::vector<cv::Point2f>(landmarks.begin() + 17, landmarks.end());
}
