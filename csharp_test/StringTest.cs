using System;
using System.Runtime.InteropServices;

namespace FacialExpressionTest
{
    class StringTest
    {
        [DllImport("FacialExpressionDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int TestFunction();

        [DllImport("FacialExpressionDLL.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int TestStringFunction(
            [MarshalAs(UnmanagedType.LPStr)] string testString
        );

        static void Main()
        {
            try 
            {
                Console.WriteLine("=== 测试字符串参数传递 ===");
                
                // 1. 测试无参数函数
                Console.WriteLine("测试1: 调用TestFunction()...");
                int result1 = TestFunction();
                Console.WriteLine($"结果: {result1}");
                
                // 2. 测试null字符串
                Console.WriteLine("\n测试2: 传递null字符串...");
                int result2 = TestStringFunction(null);
                Console.WriteLine($"结果: {result2}");
                
                // 3. 测试空字符串
                Console.WriteLine("\n测试3: 传递空字符串...");
                int result3 = TestStringFunction("");
                Console.WriteLine($"结果: {result3}");
                
                // 4. 测试普通字符串
                Console.WriteLine("\n测试4: 传递普通字符串...");
                string testStr = "Hello World";
                int result4 = TestStringFunction(testStr);
                Console.WriteLine($"字符串: '{testStr}', 长度: {result4}");
                
                // 5. 测试中文字符串
                Console.WriteLine("\n测试5: 传递中文字符串...");
                string chineseStr = "你好世界";
                int result5 = TestStringFunction(chineseStr);
                Console.WriteLine($"字符串: '{chineseStr}', 长度: {result5}");
                
                Console.WriteLine("\n所有字符串测试完成!");
                
            }
            catch (Exception ex)
            {
                Console.WriteLine($"异常: {ex.Message}");
                Console.WriteLine($"堆栈跟踪: {ex.StackTrace}");
            }
            
            Console.WriteLine("\n按任意键退出...");
            Console.ReadKey();
        }
    }
}
