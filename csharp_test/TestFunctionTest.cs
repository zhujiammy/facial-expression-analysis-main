using System;
using System.Runtime.InteropServices;

namespace FacialExpressionTest
{
    class TestFunctionTest
    {
        [DllImport("FacialExpressionDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int TestFunction();
        
        [DllImport("FacialExpressionDLL.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int TestStringFunction([MarshalAs(UnmanagedType.LPStr)] string testString);
        
        static void Main()
        {
            try 
            {
                Console.WriteLine("测试简单函数TestFunction...");
                int result = TestFunction();
                Console.WriteLine($"TestFunction返回: {result}");
                
                if (result == 42)
                {
                    Console.WriteLine("基本函数调用成功!");
                }
                
                Console.WriteLine("\n测试字符串函数TestStringFunction...");
                
                // 测试null参数
                Console.WriteLine("测试null参数:");
                int result1 = TestStringFunction(null);
                Console.WriteLine($"TestStringFunction(null) 返回: {result1}");
                
                // 测试空字符串
                Console.WriteLine("测试空字符串:");
                int result2 = TestStringFunction("");
                Console.WriteLine($"TestStringFunction(\"\") 返回: {result2}");
                
                // 测试普通字符串
                Console.WriteLine("测试普通字符串:");
                string testStr = "hello";
                int result3 = TestStringFunction(testStr);
                Console.WriteLine($"TestStringFunction(\"{testStr}\") 返回: {result3}");
                
                if (result3 == testStr.Length)
                {
                    Console.WriteLine("字符串函数调用成功!");
                }
                
            }
            catch (Exception ex)
            {
                Console.WriteLine($"异常: {ex.Message}");
                Console.WriteLine($"堆栈跟踪: {ex.StackTrace}");
            }
            
            Console.WriteLine("测试完成，按任意键退出...");
            Console.ReadKey();
        }
    }
}
