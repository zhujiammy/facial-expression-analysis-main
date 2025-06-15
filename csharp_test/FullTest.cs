using System;
using System.Runtime.InteropServices;
using System.IO;

namespace FacialExpressionTest
{
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct EmotionResultDLL
    {
        public float arousal;
        public float valence;
        public float intensity;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
        public string emotion_name;
        public int success;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
        public string error_message;
    }

    class FullTest
    {
        [DllImport("FacialExpressionDLL.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int InitializeEmotionAnalyzer(
            [MarshalAs(UnmanagedType.LPStr)] string onnxModelPath,
            [MarshalAs(UnmanagedType.LPStr)] string shapePredictorPath,
            [MarshalAs(UnmanagedType.LPStr)] string frontalizationModelPath
        );
        
        [DllImport("FacialExpressionDLL.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern EmotionResultDLL AnalyzeEmotionFromFile([MarshalAs(UnmanagedType.LPStr)] string imagePath);
        
        [DllImport("FacialExpressionDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ReleaseEmotionAnalyzer();
        
        static void Main()
        {
            Console.WriteLine("=== 完整情绪分析测试 ===\n");
            
            try
            {
                // 1. 初始化
                Console.WriteLine("1. 初始化情绪分析器...");
                int initResult = InitializeEmotionAnalyzer(
                    "model_emotion_pls30.onnx",
                    "shape_predictor_68_face_landmarks.dat", 
                    "model_frontalization.npy"
                );
                
                if (initResult != 1)
                {
                    Console.WriteLine("初始化失败!");
                    return;
                }
                Console.WriteLine("初始化成功!");
                
                // 2. 检查是否有图片文件可以测试
                Console.WriteLine("\n2. 查找测试图片...");
                string[] imageExtensions = { "*.jpg", "*.jpeg", "*.png", "*.bmp" };
                string testImagePath = null;
                
                foreach (string ext in imageExtensions)
                {
                    string[] files = Directory.GetFiles(".", ext);
                    if (files.Length > 0)
                    {
                        testImagePath = files[0];
                        break;
                    }
                }
                
                if (testImagePath == null)
                {
                    Console.WriteLine("未找到测试图片文件 (支持 jpg, jpeg, png, bmp)");
                    Console.WriteLine("请将一张包含人脸的图片放在程序目录下");
                }
                else
                {
                    Console.WriteLine($"找到测试图片: {testImagePath}");
                    
                    // 3. 分析情绪
                    Console.WriteLine("\n3. 分析情绪...");
                    EmotionResultDLL result = AnalyzeEmotionFromFile(testImagePath);
                    
                    if (result.success == 1)
                    {
                        Console.WriteLine("✅ 情绪分析成功!");
                        Console.WriteLine($"   Arousal (唤醒度): {result.arousal:F3}");
                        Console.WriteLine($"   Valence (效价): {result.valence:F3}");
                        Console.WriteLine($"   Intensity (强度): {result.intensity:F3}");
                        Console.WriteLine($"   Emotion (情绪): {result.emotion_name}");
                    }
                    else
                    {
                        Console.WriteLine("❌ 情绪分析失败:");
                        Console.WriteLine($"   错误信息: {result.error_message}");
                    }
                }
                
                // 4. 清理
                Console.WriteLine("\n4. 清理资源...");
                ReleaseEmotionAnalyzer();
                Console.WriteLine("清理完成!");
                
            }
            catch (Exception ex)
            {
                Console.WriteLine($"异常: {ex.Message}");
                Console.WriteLine($"类型: {ex.GetType().Name}");
                Console.WriteLine($"堆栈: {ex.StackTrace}");
            }
            
            Console.WriteLine("\n=== 测试完成 ===");
            Console.WriteLine("按任意键退出...");
            Console.ReadKey();
        }
    }
}
