#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "emotion_analyzer.h"
#include "model_comparison.h"
#include "facial_landmarks.h"
#include "utils.h"

void printUsage(const char* program_name) {
    std::cout << "用法: " << program_name << " [选项]\n";
    std::cout << "选项:\n";
    std::cout << "  -h, --help              显示帮助信息\n";
    std::cout << "  -i, --image <path>      分析单张图像\n";
    std::cout << "  -b, --batch <dir>       批量分析目录中的图像\n";
    std::cout << "  -c, --compare           与Python模型比较\n";
    std::cout << "  -r, --random-test       随机输入一致性测试\n";
    std::cout << "  -v, --validate          运行完整验证测试\n";
    std::cout << "  -m, --models <dir>      指定模型文件目录（默认: ../models）\n";
    std::cout << "\n示例:\n";
    std::cout << "  " << program_name << " -i ../data/images/pleased.jpg\n";
    std::cout << "  " << program_name << " -c\n";
    std::cout << "  " << program_name << " -v\n";
}

struct Config {
    std::string models_dir = "../models";
    std::string onnx_model_path;
    std::string frontalization_model_path;
    std::string shape_predictor_path;
    
    void initialize() {
        onnx_model_path = models_dir + "/model_emotion_pls30.onnx";
        frontalization_model_path = models_dir + "/model_frontalization.npy";
        shape_predictor_path = models_dir + "/shape_predictor_68_face_landmarks.dat";
    }
    
    bool validate() {
        if (!Utils::fileExists(onnx_model_path)) {
            std::cerr << "ONNX模型文件不存在: " << onnx_model_path << std::endl;
            return false;
        }
        if (!Utils::fileExists(frontalization_model_path)) {
            std::cerr << "正面化模型文件不存在: " << frontalization_model_path << std::endl;
            return false;
        }
        if (!Utils::fileExists(shape_predictor_path)) {
            std::cerr << "面部关键点检测器文件不存在: " << shape_predictor_path << std::endl;
            return false;
        }
        return true;
    }
};

void analyzeSingleImage(std::shared_ptr<EmotionAnalyzer> analyzer, const std::string& image_path) {
    std::cout << "\n========== 分析单张图像 ==========\n";
    std::cout << "图像路径: " << image_path << std::endl;
    
    if (!Utils::fileExists(image_path)) {
        std::cerr << "图像文件不存在: " << image_path << std::endl;
        return;
    }
    
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "无法读取图像: " << image_path << std::endl;
        return;
    }
    
    std::cout << "图像尺寸: " << image.cols << "x" << image.rows << std::endl;
    
    // 进行情感分析
    EmotionResult result = analyzer->analyzeEmotion(image);
    
    std::cout << "\n情感分析结果:\n";
    std::cout << "  Arousal (唤醒度): " << std::fixed << std::setprecision(6) << result.arousal << std::endl;
    std::cout << "  Valence (效价): " << std::fixed << std::setprecision(6) << result.valence << std::endl;
    std::cout << "  Intensity (强度): " << std::fixed << std::setprecision(6) << result.intensity << std::endl;
    std::cout << "  Emotion (情感): " << result.emotion_name << std::endl;
    
    // 获取并可视化面部关键点
    LandmarksData landmarks_data = analyzer->getFacialLandmarks(image);
    if (!landmarks_data.raw_landmarks.empty()) {
        std::cout << "\n检测到 " << landmarks_data.raw_landmarks.size() << " 个面部关键点\n";
        
        // 创建可视化图像
        cv::Mat vis_image = FacialLandmarks::visualizeLandmarks(image, landmarks_data.raw_landmarks);
        
        // 在图像上添加情感信息
        cv::Scalar color = Utils::emotionToColor(result.emotion_name);
        cv::putText(vis_image, result.emotion_name, 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
        
        std::string info = "A:" + std::to_string(result.arousal).substr(0, 5) + 
                          " V:" + std::to_string(result.valence).substr(0, 5) +
                          " I:" + std::to_string(result.intensity).substr(0, 5);
        cv::putText(vis_image, info, 
                   cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
        
        // 保存结果图像
        std::string output_path = "result_" + Utils::getCurrentTimeString() + ".jpg";
        std::replace(output_path.begin(), output_path.end(), ':', '-');
        std::replace(output_path.begin(), output_path.end(), ' ', '_');
        
        cv::imwrite(output_path, vis_image);
        std::cout << "结果图像已保存: " << output_path << std::endl;
        
        // 显示图像（如果支持）
        cv::imshow("Facial Expression Analysis", vis_image);
        std::cout << "按任意键继续..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    
    std::cout << "===============================\n";
}

void runModelComparison(std::shared_ptr<EmotionAnalyzer> analyzer) {
    std::cout << "\n========== 模型比较测试 ==========\n";
    
    ModelComparison comparison(analyzer);
    
    // 测试图像列表
    std::vector<std::string> test_images = {
        "../data/images/pleased.jpg",
        "../data/images/happy.jpg",
        "../data/images/sad.jpg"
    };
    
    // 过滤存在的图像
    std::vector<std::string> valid_images;
    for (const auto& img : test_images) {
        if (Utils::fileExists(img)) {
            valid_images.push_back(img);
        } else {
            std::cout << "跳过不存在的图像: " << img << std::endl;
        }
    }
      if (valid_images.empty()) {
        std::cerr << "No valid test images found" << std::endl;
        return;
    }
    
    // 执行比较
    ComparisonResult result = comparison.compareWithPythonModel("../source/compare_with_cpp.py", valid_images);
      // 打印结果
    comparison.printComparisonResults(result);
    
    std::cout << "===============================\n";
}

void runRandomTest(std::shared_ptr<EmotionAnalyzer> analyzer) {
    std::cout << "\n========== 随机输入一致性测试 ==========\n";
    
    ModelComparison comparison(analyzer);
    
    ComparisonResult result = comparison.testRandomInputConsistency(10, 1275);
    
    comparison.printComparisonResults(result);
    
    std::cout << "===============================\n";
}

void runFullValidation(std::shared_ptr<EmotionAnalyzer> analyzer) {
    std::cout << "\n========== 完整验证测试 ==========\n";
    
    ModelComparison comparison(analyzer);
    
    std::string output_dir = "validation_results_" + Utils::getCurrentTimeString();
    std::replace(output_dir.begin(), output_dir.end(), ':', '-');
    std::replace(output_dir.begin(), output_dir.end(), ' ', '_');
    
    bool success = comparison.runFullValidationTest(output_dir);
    
    if (success) {
        std::cout << "\n🎉 所有验证测试通过！C++模型与Python模型一致。\n";
    } else {
        std::cout << "\n⚠️  部分验证测试失败，请检查详细报告。\n";
    }
    
    std::cout << "===============================\n";
}

int main(int argc, char* argv[]) {
    std::cout << "========================================\n";
    std::cout << "   面部表情分析 - C++ ONNX 模型测试\n";
    std::cout << "========================================\n";
    
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    // 解析命令行参数
    Config config;
    std::string mode;
    std::string image_path;
    std::string batch_dir;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-i" || arg == "--image") {
            if (i + 1 < argc) {
                mode = "single";
                image_path = argv[++i];
            } else {
                std::cerr << "错误: -i 选项需要指定图像路径\n";
                return 1;
            }
        } else if (arg == "-b" || arg == "--batch") {
            if (i + 1 < argc) {
                mode = "batch";
                batch_dir = argv[++i];
            } else {
                std::cerr << "错误: -b 选项需要指定目录路径\n";
                return 1;
            }
        } else if (arg == "-c" || arg == "--compare") {
            mode = "compare";
        } else if (arg == "-r" || arg == "--random-test") {
            mode = "random";
        } else if (arg == "-v" || arg == "--validate") {
            mode = "validate";
        } else if (arg == "-m" || arg == "--models") {
            if (i + 1 < argc) {
                config.models_dir = argv[++i];
            } else {
                std::cerr << "错误: -m 选项需要指定模型目录\n";
                return 1;
            }
        }
    }
    
    if (mode.empty()) {
        std::cerr << "错误: 必须指定操作模式\n";
        printUsage(argv[0]);
        return 1;
    }
    
    // 初始化配置
    config.initialize();
    
    std::cout << "模型文件配置:\n";
    std::cout << "  ONNX模型: " << config.onnx_model_path << std::endl;
    std::cout << "  正面化模型: " << config.frontalization_model_path << std::endl;
    std::cout << "  关键点检测器: " << config.shape_predictor_path << std::endl;
    
    if (!config.validate()) {
        std::cerr << "模型文件验证失败，请检查文件路径\n";
        return 1;
    }
    
    // 初始化情感分析器
    std::cout << "\n初始化情感分析器...\n";
    auto analyzer = std::make_shared<EmotionAnalyzer>(
        config.onnx_model_path,
        config.frontalization_model_path,
        config.shape_predictor_path
    );
    
    if (!analyzer->initialize()) {
        std::cerr << "情感分析器初始化失败\n";
        return 1;
    }
    
    // 执行对应的操作
    try {
        if (mode == "single") {
            analyzeSingleImage(analyzer, image_path);
        } else if (mode == "compare") {
            runModelComparison(analyzer);
        } else if (mode == "random") {
            runRandomTest(analyzer);
        } else if (mode == "validate") {
            runFullValidation(analyzer);
        } else {
            std::cerr << "未知的操作模式: " << mode << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "执行过程中发生异常: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n程序执行完成。\n";
    return 0;
}
