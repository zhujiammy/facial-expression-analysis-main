#pragma once

#include <vector>
#include <string>
#include <memory>
#include "emotion_analyzer.h"

struct ComparisonResult {
    bool success;
    float max_difference;
    float mean_difference;
    std::vector<std::vector<float>> cpp_predictions;
    std::vector<std::vector<float>> python_predictions;
    std::vector<std::vector<float>> differences;
    std::string error_message;
};

class ModelComparison {
public:
    ModelComparison(std::shared_ptr<EmotionAnalyzer> analyzer);
    
    // 与Python joblib模型比较
    ComparisonResult compareWithPythonModel(const std::string& python_script_path,
                                          const std::vector<std::string>& test_images);
    
    // 测试随机输入的一致性
    ComparisonResult testRandomInputConsistency(int num_samples = 10, 
                                              int feature_dims = 1275);
    
    // 加载Python预测结果
    std::vector<std::vector<float>> loadPythonPredictions(const std::string& results_file);
    
    // 保存C++预测结果
    bool saveCppPredictions(const std::vector<std::vector<float>>& predictions,
                          const std::string& output_file);
    
    // 计算预测差异统计
    void calculateDifferenceStats(const std::vector<std::vector<float>>& cpp_pred,
                                const std::vector<std::vector<float>>& python_pred,
                                ComparisonResult& result);
    
    // 生成测试报告
    std::string generateReport(const ComparisonResult& result);
    
    // 运行完整的模型验证测试
    bool runFullValidationTest(const std::string& output_dir);

private:
    std::shared_ptr<EmotionAnalyzer> analyzer_;    
    // 辅助函数
    bool executePythonScript(const std::string& script_path, 
                           const std::vector<std::string>& args);
    
    std::vector<float> generateRandomFeatureVector(int dims);

public:
    void printComparisonResults(const ComparisonResult& result);
};
