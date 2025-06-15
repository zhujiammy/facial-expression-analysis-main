#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "Facial Expression Analysis Test" << std::endl;
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
    
    // Test OpenCV
    cv::Mat test_image = cv::Mat::zeros(100, 100, CV_8UC3);
    if (!test_image.empty()) {
        std::cout << "OpenCV working correctly" << std::endl;
    }
    
    return 0;
}
