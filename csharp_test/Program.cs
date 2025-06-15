using System;
using System.IO;

namespace FacialExpressionAnalysis
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== 面部表情分析 C# 测试程序 ===");
            
            try
            {
                // 模型文件路径（相对于DLL目录）
                string onnxModelPath = "model_emotion_pls30.onnx";
                string shapePredictorPath = "shape_predictor_68_face_landmarks.dat";
                string frontalizationModelPath = "model_frontalization.npy";
                
                Console.WriteLine("正在初始化情绪分析器...");
                
                // 初始化分析器
                int initResult = FacialExpressionAPI.InitializeEmotionAnalyzer(
                    onnxModelPath, 
                    shapePredictorPath, 
                    frontalizationModelPath
                );
                
                if (initResult == 0)
                {
                    string error = FacialExpressionAPI.GetLastError();
                    Console.WriteLine($"初始化失败: {error}");
                    return;
                }
                
                Console.WriteLine("初始化成功！");
                  // 测试图片路径
                string[] testImages = {
                    "../data/images/pleased.jpg",
                    "../data/images/angry.jpg",
                    "../data/images/happy.jpg",
                    "../data/images/sad.jpg"
                };
                
                foreach (string imagePath in testImages)
                {
                    Console.WriteLine($"\n分析图片: {imagePath}");
                    
                    if (!File.Exists(imagePath))
                    {
                        Console.WriteLine($"图片文件不存在: {imagePath}");
                        continue;
                    }
                    
                    // 分析情绪
                    EmotionResult result = FacialExpressionAPI.AnalyzeEmotionFromFile(imagePath);
                    
                    if (result.Success == 1)
                    {
                        Console.WriteLine($"  Arousal: {result.Arousal:F6}");
                        Console.WriteLine($"  Valence: {result.Valence:F6}");
                        Console.WriteLine($"  Intensity: {result.Intensity:F6}");
                        Console.WriteLine($"  情绪: {result.EmotionName}");
                    }
                    else
                    {
                        Console.WriteLine($"  分析失败: {result.ErrorMessage}");
                    }
                }
                
                // 测试字节数组输入（读取一个图片文件）
                string testImagePath = "../data/images/pleased.jpg";
                if (File.Exists(testImagePath))
                {
                    Console.WriteLine($"\n测试字节数组输入: {testImagePath}");
                    
                    byte[] imageBytes = File.ReadAllBytes(testImagePath);
                    
                    // 注意：这里简化了处理，实际应用中需要正确解析图片的宽高和通道数
                    // 对于从文件读取的情况，可以传递0让DLL内部处理解码
                    EmotionResult result = FacialExpressionAPI.AnalyzeEmotionFromBytes(
                        imageBytes, imageBytes.Length, 0, 0, 0
                    );
                    
                    if (result.Success == 1)
                    {
                        Console.WriteLine($"  Arousal: {result.Arousal:F6}");
                        Console.WriteLine($"  Valence: {result.Valence:F6}");
                        Console.WriteLine($"  Intensity: {result.Intensity:F6}");
                        Console.WriteLine($"  情绪: {result.EmotionName}");
                    }
                    else
                    {
                        Console.WriteLine($"  分析失败: {result.ErrorMessage}");
                    }
                }
                
                Console.WriteLine("\n测试完成，正在释放资源...");
                FacialExpressionAPI.ReleaseEmotionAnalyzer();
                Console.WriteLine("资源释放完成！");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"程序异常: {ex.Message}");
                Console.WriteLine($"异常详情: {ex.StackTrace}");
            }
            
            Console.WriteLine("\n按任意键退出...");
            Console.ReadKey();
        }
    }
}
