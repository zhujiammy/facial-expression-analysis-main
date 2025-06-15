using System;
using System.Runtime.InteropServices;
using System.IO;

namespace FacialExpressionTest
{
    class DiagnosticTest
    {
        // 先测试最基本的函数
        [DllImport("FacialExpressionDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int TestFunction();
        
        [DllImport("FacialExpressionDLL.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int TestStringFunction([MarshalAs(UnmanagedType.LPStr)] string test_string);
        
        // 包装初始化函数用于异常捕获
        [DllImport("FacialExpressionDLL.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int InitializeEmotionAnalyzer(
            [MarshalAs(UnmanagedType.LPStr)] string onnxModelPath,
            [MarshalAs(UnmanagedType.LPStr)] string shapePredictorPath,
            [MarshalAs(UnmanagedType.LPStr)] string frontalizationModelPath
        );
        
        [DllImport("FacialExpressionDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ReleaseEmotionAnalyzer();
        
        [DllImport("FacialExpressionDLL.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern IntPtr GetLastError();
        
        static void Main()
        {
            Console.WriteLine("=== 诊断测试开始 ===\n");
            
            // 检查文件是否存在
            Console.WriteLine("1. 检查DLL和依赖文件是否存在:");
            string[] requiredFiles = {
                "FacialExpressionDLL.dll",
                "onnxruntime.dll",
                "model_emotion_pls30.onnx",
                "shape_predictor_68_face_landmarks.dat",
                "model_frontalization.npy"
            };
            
            bool allFilesExist = true;
            foreach (string file in requiredFiles)
            {
                bool exists = File.Exists(file);
                Console.WriteLine($"   {file}: {(exists ? "存在" : "缺失")}");
                if (!exists) allFilesExist = false;
            }
            
            if (!allFilesExist)
            {
                Console.WriteLine("\n错误: 缺少必要文件，请检查文件复制");
                Console.ReadKey();
                return;
            }
            
            Console.WriteLine("\n2. 测试基本函数调用:");
            try
            {
                int result = TestFunction();
                Console.WriteLine($"   TestFunction() = {result} (预期: 42)");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   TestFunction() 失败: {ex.Message}");
                Console.ReadKey();
                return;
            }
            
            Console.WriteLine("\n3. 测试字符串参数:");
            try
            {
                int length = TestStringFunction("hello");
                Console.WriteLine($"   TestStringFunction(\"hello\") = {length} (预期: 5)");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   TestStringFunction() 失败: {ex.Message}");
                Console.ReadKey();
                return;
            }
            
            Console.WriteLine("\n4. 测试初始化（空参数）:");
            try
            {
                int result = InitializeEmotionAnalyzer(null, null, null);
                Console.WriteLine($"   InitializeEmotionAnalyzer(null, null, null) = {result}");
                
                // 获取错误信息
                IntPtr errorPtr = GetLastError();
                if (errorPtr != IntPtr.Zero)
                {
                    string error = Marshal.PtrToStringAnsi(errorPtr) ?? "";
                    Console.WriteLine($"   错误信息: {error}");
                }
                
                if (result == 1)
                {
                    Console.WriteLine("   清理资源...");
                    ReleaseEmotionAnalyzer();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   InitializeEmotionAnalyzer(null) 失败: {ex.Message}");
                Console.WriteLine($"   异常类型: {ex.GetType().Name}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"   内部异常: {ex.InnerException.Message}");
                }
            }
            
            Console.WriteLine("\n5. 测试初始化（有效路径）:");
            try
            {
                string onnxPath = "model_emotion_pls30.onnx";
                string shapePath = "shape_predictor_68_face_landmarks.dat";
                string frontPath = "model_frontalization.npy";
                
                Console.WriteLine($"   路径验证:");
                Console.WriteLine($"     ONNX: {File.Exists(onnxPath)}");
                Console.WriteLine($"     Shape: {File.Exists(shapePath)}");
                Console.WriteLine($"     Front: {File.Exists(frontPath)}");
                
                int result = InitializeEmotionAnalyzer(onnxPath, shapePath, frontPath);
                Console.WriteLine($"   InitializeEmotionAnalyzer(有效路径) = {result}");
                
                // 获取错误信息
                IntPtr errorPtr = GetLastError();
                if (errorPtr != IntPtr.Zero)
                {
                    string error = Marshal.PtrToStringAnsi(errorPtr) ?? "";
                    Console.WriteLine($"   错误信息: {error}");
                }
                
                if (result == 1)
                {
                    Console.WriteLine("   初始化成功! 清理资源...");
                    ReleaseEmotionAnalyzer();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   InitializeEmotionAnalyzer(有效路径) 失败: {ex.Message}");
                Console.WriteLine($"   异常类型: {ex.GetType().Name}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"   内部异常: {ex.InnerException.Message}");
                }
            }
            
            Console.WriteLine("\n=== 诊断测试完成 ===");
            Console.WriteLine("按任意键退出...");
            Console.ReadKey();
        }
    }
}
