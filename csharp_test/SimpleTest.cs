using System;
using System.Runtime.InteropServices;

namespace FacialExpressionTest
{
    class SimpleTest
    {
        [DllImport("FacialExpressionDLL.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern IntPtr GetLastError();
        
        static void Main()
        {
            try 
            {
                Console.WriteLine("测试GetLastError函数...");
                IntPtr errorPtr = GetLastError();
                Console.WriteLine($"GetLastError返回指针: {errorPtr}");
                
                if (errorPtr != IntPtr.Zero)
                {
                    string error = Marshal.PtrToStringAnsi(errorPtr);
                    Console.WriteLine($"错误信息: {error}");
                }
                else
                {
                    Console.WriteLine("无错误信息 (指针为空)");
                }
                
                Console.WriteLine("GetLastError测试成功!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"异常: {ex.Message}");
                Console.WriteLine($"堆栈跟踪: {ex.StackTrace}");
            }
            
            Console.WriteLine("测试完成");
            Console.ReadKey();
        }
    }
}
