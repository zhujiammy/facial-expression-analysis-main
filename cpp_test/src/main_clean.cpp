#include "emotion_analyzer.h"
#include "model_comparison.h"
#include "utils.h"
#include <iostream>
#include <string>
#include <vector>

void showHelp(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help              Show help information\n";
    std::cout << "  -i, --image <path>      Analyze single image\n";
    std::cout << "  -b, --batch <dir>       Batch analyze images in directory\n";
    std::cout << "  -c, --compare           Compare with Python model\n";
    std::cout << "  -v, --verbose           Verbose output\n";
    std::cout << "  --model-path <path>     Path to ONNX model\n";
    std::cout << "  --shape-predictor <path> Path to shape predictor\n";
    std::cout << "  --frontalization <path> Path to frontalization model\n";
}

void analyzeImage(const std::string& image_path, EmotionAnalyzer& analyzer) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Cannot load image " << image_path << std::endl;
        return;
    }
    
    std::cout << "Analyzing image: " << image_path << std::endl;
    
    auto result = analyzer.predictEmotion(image);
    
    if (result.success) {
        std::cout << "Predicted emotion: " << result.predicted_emotion << std::endl;
        std::cout << "Confidence: " << result.confidence << std::endl;
        
        std::cout << "All probabilities:" << std::endl;
        auto labels = analyzer.getEmotionLabels();
        for (size_t i = 0; i < labels.size() && i < result.probabilities.size(); ++i) {
            std::cout << "  " << labels[i] << ": " << result.probabilities[i] << std::endl;
        }
    } else {
        std::cerr << "Analysis failed: " << result.error_message << std::endl;
    }
}

void batchAnalyze(const std::string& directory_path, EmotionAnalyzer& analyzer) {
    std::cout << "Batch analyzing images in: " << directory_path << std::endl;
    
    // Implementation would scan directory and analyze each image
    std::vector<std::string> image_files = getImageFiles(directory_path);
    
    for (const auto& file : image_files) {
        analyzeImage(file, analyzer);
    }
}

void compareModels() {
    std::cout << "===============================\n";
    std::cout << "Starting model comparison tests\n";
    std::cout << "===============================\n";
    
    ModelComparison comparison;
    
    // Test images
    std::vector<std::string> test_images = {
        "../data/images/happy.jpg",
        "../data/images/sad.jpg",
        "../data/images/angry.jpg",
        "../data/images/example.png"
    };
    
    // Validate images exist
    std::vector<std::string> valid_images;
    for (const auto& img : test_images) {
        cv::Mat test_img = cv::imread(img);
        if (!test_img.empty()) {
            valid_images.push_back(img);
        } else {
            std::cout << "Warning: Cannot load " << img << std::endl;
        }
    }
    
    if (valid_images.empty()) {
        std::cerr << "No valid test images found" << std::endl;
        return;
    }
    
    // Execute comparison
    ComparisonResult result = comparison.compareWithPythonModel("../source/compare_with_cpp.py", valid_images);
    
    // Print results
    comparison.printComparisonResults(result);
    
    std::cout << "===============================\n";
}

int main(int argc, char* argv[]) {
    // Default paths
    std::string model_path = "model_emotion_pls30.onnx";
    std::string shape_predictor_path = "shape_predictor_68_face_landmarks.dat";
    std::string frontalization_path = "model_frontalization.npy";
    
    // Parse command line arguments
    bool compare_mode = false;
    bool verbose = false;
    std::string image_path;
    std::string batch_directory;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            showHelp(argv[0]);
            return 0;
        } else if (arg == "-c" || arg == "--compare") {
            compare_mode = true;
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "-i" || arg == "--image") {
            if (i + 1 < argc) {
                image_path = argv[++i];
            } else {
                std::cerr << "Error: --image requires a path" << std::endl;
                return 1;
            }
        } else if (arg == "-b" || arg == "--batch") {
            if (i + 1 < argc) {
                batch_directory = argv[++i];
            } else {
                std::cerr << "Error: --batch requires a directory path" << std::endl;
                return 1;
            }
        } else if (arg == "--model-path") {
            if (i + 1 < argc) {
                model_path = argv[++i];
            }
        } else if (arg == "--shape-predictor") {
            if (i + 1 < argc) {
                shape_predictor_path = argv[++i];
            }
        } else if (arg == "--frontalization") {
            if (i + 1 < argc) {
                frontalization_path = argv[++i];
            }
        }
    }
    
    if (compare_mode) {
        compareModels();
        return 0;
    }
    
    // Initialize emotion analyzer
    EmotionAnalyzer analyzer(model_path, frontalization_path, shape_predictor_path);
    
    if (!analyzer.initialize()) {
        std::cerr << "Failed to initialize emotion analyzer" << std::endl;
        return 1;
    }
    
    // Execute based on mode
    if (!image_path.empty()) {
        analyzeImage(image_path, analyzer);
    } else if (!batch_directory.empty()) {
        batchAnalyze(batch_directory, analyzer);
    } else {
        // Default behavior - compare models
        compareModels();
    }
    
    return 0;
}
