using System;
using System.Runtime.InteropServices;

namespace FacialExpressionTest
{
    class InitTest
    {
        [DllImport("FacialExpressionDLL.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern IntPtr GetLastError();
        
        [DllImport("FacialExpressionDLL.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int InitializeEmotionAnalyzer(
            [MarshalAs(UnmanagedType.LPStr)] string onnxModelPath,
            [MarshalAs(UnmanagedType.LPStr)] string shapePredictorPath,
            [MarshalAs(UnmanagedType.LPStr)] string frontalizationModelPath
        );
        
        [DllImport("FacialExpressionDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ReleaseEmotionAnalyzer();
        
        static string GetErrorMessage()
        {
            IntPtr errorPtr = GetLastError();
            if (errorPtr != IntPtr.Zero)
            {
                return Marshal.PtrToStringAnsi(errorPtr) ?? "";
            }
            return "";
        }
        
        static void Main()
        {
            try 
            {
                Console.WriteLine("=== 测试InitializeEmotionAnalyzer ===");
                
                // 测试空指针参数
                Console.WriteLine("测试1: 传递null参数...");
                int result1 = InitializeEmotionAnalyzer(null, null, null);
                Console.WriteLine($"结果: {result1}");
                Console.WriteLine($"错误信息: {GetErrorMessage()}");
                
                Console.WriteLine("\n测试2: 传递空字符串...");
                int result2 = InitializeEmotionAnalyzer("", "", "");
                Console.WriteLine($"结果: {result2}");
                Console.WriteLine($"错误信息: {GetErrorMessage()}");
                
                Console.WriteLine("\n测试3: 传递正确的模型路径...");
                
                // 首先复制模型文件到当前目录
                string onnxPath = "model_emotion_pls30.onnx";
                string shapePath = "shape_predictor_68_face_landmarks.dat";
                string frontPath = "model_frontalization.npy";
                
                Console.WriteLine($"尝试初始化，路径:");
                Console.WriteLine($"  ONNX: {onnxPath}");
                Console.WriteLine($"  Shape: {shapePath}");
                Console.WriteLine($"  Front: {frontPath}");
                
                int result3 = InitializeEmotionAnalyzer(onnxPath, shapePath, frontPath);
                Console.WriteLine($"结果: {result3}");
                
                string error = GetErrorMessage();
                if (!string.IsNullOrEmpty(error))
                {
                    Console.WriteLine($"错误信息: {error}");
                }
                
                if (result3 == 1)
                {
                    Console.WriteLine("初始化成功!");
                    ReleaseEmotionAnalyzer();
                    Console.WriteLine("资源已释放");
                }
                
            }
            catch (Exception ex)
            {
                Console.WriteLine($"异常: {ex.Message}");
                Console.WriteLine($"堆栈跟踪: {ex.StackTrace}");
            }
            
            Console.WriteLine("\n测试完成，按任意键退出...");
            Console.ReadKey();
        }
    }
}
