using System;
using System.Runtime.InteropServices;

namespace FacialExpressionAnalysis
{
    [StructLayout(LayoutKind.Sequential)]
    public struct EmotionResult
    {
        public float Arousal;
        public float Valence;
        public float Intensity;
        
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
        public string EmotionName;
        
        public int Success;
        
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
        public string ErrorMessage;
    }

    public static class FacialExpressionAPI
    {
        private const string DLL_NAME = "FacialExpressionDLL.dll";

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int InitializeEmotionAnalyzer(
            [MarshalAs(UnmanagedType.LPStr)] string onnxModelPath,
            [MarshalAs(UnmanagedType.LPStr)] string shapePredictorPath,
            [MarshalAs(UnmanagedType.LPStr)] string frontalizationModelPath
        );

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern EmotionResult AnalyzeEmotionFromFile(
            [MarshalAs(UnmanagedType.LPStr)] string imagePath
        );

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern EmotionResult AnalyzeEmotionFromBytes(
            byte[] imageData,
            int dataLength,
            int width,
            int height,
            int channels
        );

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ReleaseEmotionAnalyzer();

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string GetLastError();
    }
}
