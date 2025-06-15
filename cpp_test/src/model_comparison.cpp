#include "model_comparison.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <random>
#include <iomanip>

ModelComparison::ModelComparison(std::shared_ptr<EmotionAnalyzer> analyzer)
    : analyzer_(analyzer) {
}

ComparisonResult ModelComparison::compareWithPythonModel(const std::string& python_script_path,
                                                       const std::vector<std::string>& test_images) {
    ComparisonResult result;
    result.success = false;
    
    try {
        std::cout << "开始与Python模型比较..." << std::endl;
        
        // 1. 生成测试数据并保存C++预测结果
        std::vector<std::vector<float>> cpp_predictions;
        
        for (const auto& image_path : test_images) {
            if (!Utils::fileExists(image_path)) {
                std::cerr << "图像文件不存在: " << image_path << std::endl;
                continue;
            }
            
            cv::Mat image = cv::imread(image_path);
            if (image.empty()) {
                std::cerr << "无法读取图像: " << image_path << std::endl;
                continue;
            }            EmotionResult emotion_result = analyzer_->analyzeEmotion(image);
            std::vector<float> prediction = {emotion_result.arousal, emotion_result.valence};
            cpp_predictions.push_back(prediction);
            
            std::cout << "C++预测结果 - " << Utils::getFileExtension(image_path) 
                     << ": arousal=" << std::fixed << std::setprecision(6) << emotion_result.arousal 
                     << ", valence=" << std::fixed << std::setprecision(6) << emotion_result.valence 
                     << ", intensity=" << std::fixed << std::setprecision(6) << emotion_result.intensity 
                     << ", emotion=" << emotion_result.emotion_name << std::endl;
        }
        
        // 2. 保存C++结果到文件
        std::string cpp_results_file = "cpp_predictions.txt";
        if (!saveCppPredictions(cpp_predictions, cpp_results_file)) {
            result.error_message = "无法保存C++预测结果";
            return result;
        }
        
        // 3. 执行Python脚本进行比较
        std::vector<std::string> python_args;
        for (const auto& img : test_images) {
            python_args.push_back(img);
        }
        
        if (!executePythonScript(python_script_path, python_args)) {
            result.error_message = "Python脚本执行失败";
            return result;
        }
        
        // 4. 加载Python预测结果
        std::string python_results_file = "python_predictions.txt";
        std::vector<std::vector<float>> python_predictions = loadPythonPredictions(python_results_file);
        
        if (python_predictions.empty()) {
            result.error_message = "无法加载Python预测结果";
            return result;
        }
        
        // 5. 计算差异统计
        result.cpp_predictions = cpp_predictions;
        result.python_predictions = python_predictions;
        calculateDifferenceStats(cpp_predictions, python_predictions, result);
        
        result.success = true;
        std::cout << "模型比较完成" << std::endl;
        
    } catch (const std::exception& e) {
        result.error_message = std::string("比较过程中发生异常: ") + e.what();
        std::cerr << result.error_message << std::endl;
    }
    
    return result;
}

ComparisonResult ModelComparison::testRandomInputConsistency(int num_samples, int feature_dims) {
    ComparisonResult result;
    result.success = false;
    
    try {
        std::cout << "测试随机输入一致性..." << std::endl;
        std::cout << "样本数: " << num_samples << ", 特征维度: " << feature_dims << std::endl;
        
        std::vector<std::vector<float>> cpp_predictions;
        std::vector<std::vector<float>> test_features;
        
        // 生成随机测试数据
        for (int i = 0; i < num_samples; ++i) {
            std::vector<float> features = generateRandomFeatureVector(feature_dims);
            test_features.push_back(features);
            
            // 使用C++模型预测
            std::vector<float> prediction = analyzer_->predictWithONNX(features);
            cpp_predictions.push_back(prediction);
        }
        
        // 保存测试数据和C++结果
        std::ofstream features_file("test_features.txt");
        std::ofstream cpp_file("cpp_random_predictions.txt");
        
        for (size_t i = 0; i < test_features.size(); ++i) {
            // 保存特征
            for (size_t j = 0; j < test_features[i].size(); ++j) {
                features_file << test_features[i][j];
                if (j < test_features[i].size() - 1) features_file << ",";
            }
            features_file << "\n";
            
            // 保存C++预测结果
            for (size_t j = 0; j < cpp_predictions[i].size(); ++j) {
                cpp_file << cpp_predictions[i][j];
                if (j < cpp_predictions[i].size() - 1) cpp_file << ",";
            }
            cpp_file << "\n";
        }
        
        features_file.close();
        cpp_file.close();
        
        // 执行Python脚本进行相同的预测
        std::vector<std::string> args = {"test_features.txt"};
        if (!executePythonScript("compare_random_predictions.py", args)) {
            result.error_message = "Python随机预测脚本执行失败";
            return result;
        }
        
        // 加载Python结果
        std::vector<std::vector<float>> python_predictions = loadPythonPredictions("python_random_predictions.txt");
          if (python_predictions.size() != cpp_predictions.size()) {
            result.error_message = "Mismatch between C++ and Python prediction counts";
            return result;
        }
        
        // 计算统计信息
        result.cpp_predictions = cpp_predictions;
        result.python_predictions = python_predictions;
        calculateDifferenceStats(cpp_predictions, python_predictions, result);
        
        result.success = true;
        std::cout << "Random input consistency test completed" << std::endl;
        
    } catch (const std::exception& e) {
        result.error_message = std::string("一致性测试中发生异常: ") + e.what();
        std::cerr << result.error_message << std::endl;
    }
    
    return result;
}

std::vector<std::vector<float>> ModelComparison::loadPythonPredictions(const std::string& results_file) {
    std::vector<std::vector<float>> predictions;
    
    std::ifstream file(results_file);
    if (!file.is_open()) {
        std::cerr << "无法打开Python结果文件: " << results_file << std::endl;
        return predictions;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> prediction;
        std::stringstream ss(line);
        std::string value;
        
        while (std::getline(ss, value, ',')) {
            try {
                prediction.push_back(std::stof(value));
            } catch (const std::exception& e) {
                std::cerr << "解析Python结果时出错: " << value << std::endl;
            }
        }
        
        if (!prediction.empty()) {
            predictions.push_back(prediction);
        }
    }
    
    file.close();
    return predictions;
}

bool ModelComparison::saveCppPredictions(const std::vector<std::vector<float>>& predictions,
                                       const std::string& output_file) {
    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cerr << "无法创建C++结果文件: " << output_file << std::endl;
        return false;
    }
    
    for (const auto& prediction : predictions) {
        for (size_t i = 0; i < prediction.size(); ++i) {
            file << prediction[i];
            if (i < prediction.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    
    file.close();
    return true;
}

void ModelComparison::calculateDifferenceStats(const std::vector<std::vector<float>>& cpp_pred,
                                              const std::vector<std::vector<float>>& python_pred,
                                              ComparisonResult& result) {
    if (cpp_pred.size() != python_pred.size()) {
        result.error_message = "预测结果数量不匹配";
        return;
    }
    
    result.differences.clear();
    std::vector<float> all_diffs;
    
    for (size_t i = 0; i < cpp_pred.size(); ++i) {
        if (cpp_pred[i].size() != python_pred[i].size()) {
            result.error_message = "预测维度不匹配";
            return;
        }
        
        std::vector<float> sample_diff;
        for (size_t j = 0; j < cpp_pred[i].size(); ++j) {
            float diff = std::abs(cpp_pred[i][j] - python_pred[i][j]);
            sample_diff.push_back(diff);
            all_diffs.push_back(diff);
        }
        result.differences.push_back(sample_diff);
    }
    
    if (!all_diffs.empty()) {
        result.max_difference = *std::max_element(all_diffs.begin(), all_diffs.end());
        result.mean_difference = Utils::calculateMean(all_diffs);
    }
}

std::string ModelComparison::generateReport(const ComparisonResult& result) {
    std::stringstream report;
    
    report << "========== 模型比较报告 ==========\n";
    report << "时间: " << Utils::getCurrentTimeString() << "\n\n";
    
    if (!result.success) {
        report << "❌ 比较失败: " << result.error_message << "\n";
        return report.str();
    }
    
    report << "✅ 比较成功\n";
    report << "测试样本数: " << result.cpp_predictions.size() << "\n";
    report << "最大差异: " << std::fixed << std::setprecision(8) << result.max_difference << "\n";
    report << "平均差异: " << std::fixed << std::setprecision(8) << result.mean_difference << "\n\n";
    
    // 详细结果对比
    report << "详细预测对比:\n";
    for (size_t i = 0; i < std::min(result.cpp_predictions.size(), size_t(5)); ++i) {
        report << "样本 " << (i + 1) << ":\n";
        report << "  C++:    [";
        for (size_t j = 0; j < result.cpp_predictions[i].size(); ++j) {
            report << std::fixed << std::setprecision(6) << result.cpp_predictions[i][j];
            if (j < result.cpp_predictions[i].size() - 1) report << ", ";
        }
        report << "]\n";
        
        report << "  Python: [";
        for (size_t j = 0; j < result.python_predictions[i].size(); ++j) {
            report << std::fixed << std::setprecision(6) << result.python_predictions[i][j];
            if (j < result.python_predictions[i].size() - 1) report << ", ";
        }
        report << "]\n";
        
        report << "  差异:   [";
        for (size_t j = 0; j < result.differences[i].size(); ++j) {
            report << std::fixed << std::setprecision(6) << result.differences[i][j];
            if (j < result.differences[i].size() - 1) report << ", ";
        }
        report << "]\n\n";
    }
    
    // 一致性评估
    if (result.max_difference < 1e-5) {
        report << "🎉 结论: 模型预测高度一致，转换成功！\n";
    } else if (result.max_difference < 1e-3) {
        report << "✅ 结论: 模型预测基本一致，存在微小数值差异\n";
    } else {
        report << "Warning: Large prediction differences detected, check conversion process\n";
    }
    
    report << "=====================================\n";
    
    return report.str();
}

bool ModelComparison::executePythonScript(const std::string& script_path, 
                                        const std::vector<std::string>& args) {
    std::stringstream cmd;
    cmd << "python " << script_path;
    
    for (const auto& arg : args) {
        cmd << " \"" << arg << "\"";
    }
    
    std::cout << "执行命令: " << cmd.str() << std::endl;
    
    int result = std::system(cmd.str().c_str());
    return result == 0;
}

std::vector<float> ModelComparison::generateRandomFeatureVector(int dims) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<float> dis(0.0f, 1.0f);
    
    std::vector<float> features;
    features.reserve(dims);
    
    for (int i = 0; i < dims; ++i) {
        features.push_back(dis(gen));
    }
    
    return features;
}

void ModelComparison::printComparisonResults(const ComparisonResult& result) {
    std::cout << generateReport(result) << std::endl;
}

bool ModelComparison::runFullValidationTest(const std::string& output_dir) {
    std::cout << "运行完整验证测试..." << std::endl;
    
    // 创建输出目录
    std::string mkdir_cmd = "mkdir -p " + output_dir;
    std::system(mkdir_cmd.c_str());
    
    bool all_passed = true;
    
    // 1. 随机输入一致性测试
    std::cout << "\n1. 随机输入一致性测试" << std::endl;
    ComparisonResult random_result = testRandomInputConsistency(10, 1275);
    
    std::string random_report = generateReport(random_result);
    std::ofstream random_file(output_dir + "/random_test_report.txt");
    random_file << random_report;
    random_file.close();
    
    if (!random_result.success || random_result.max_difference > 1e-5) {
        all_passed = false;
        std::cout << "❌ 随机输入测试失败" << std::endl;
    } else {
        std::cout << "✅ 随机输入测试通过" << std::endl;
    }
    
    // 2. 真实图像测试
    std::cout << "\n2. 真实图像测试" << std::endl;
    std::vector<std::string> test_images = {
        "../data/images/pleased.jpg",
        "../data/images/happy.jpg",
        "../data/images/sad.jpg"
    };
    
    ComparisonResult image_result = compareWithPythonModel("compare_image_predictions.py", test_images);
    
    std::string image_report = generateReport(image_result);
    std::ofstream image_file(output_dir + "/image_test_report.txt");
    image_file << image_report;
    image_file.close();
    
    if (!image_result.success || image_result.max_difference > 1e-3) {
        all_passed = false;
        std::cout << "❌ 图像测试失败" << std::endl;
    } else {
        std::cout << "✅ 图像测试通过" << std::endl;
    }
    
    // 生成总结报告
    std::ofstream summary_file(output_dir + "/validation_summary.txt");
    summary_file << "========== 验证测试总结 ==========\n";
    summary_file << "时间: " << Utils::getCurrentTimeString() << "\n\n";
    summary_file << "随机输入测试: " << (random_result.success ? "通过" : "失败") << "\n";
    summary_file << "图像预测测试: " << (image_result.success ? "通过" : "失败") << "\n\n";
    summary_file << "总体结果: " << (all_passed ? "✅ 所有测试通过" : "❌ 存在失败的测试") << "\n";
    summary_file << "=====================================\n";
    summary_file.close();
    
    std::cout << "\n" << (all_passed ? "All validation tests passed!" : "Some tests failed, please check the report") << std::endl;
    std::cout << "详细报告保存至: " << output_dir << std::endl;
    
    return all_passed;
}
