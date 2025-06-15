#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取npy文件信息

@author: 查看脚本
@date: 2025-06-14
"""

import numpy as np
import os

def read_npy_file_info(file_path):
    """
    读取npy文件并显示详细信息
    
    Args:
        file_path (str): npy文件路径
    """
    print(f"正在读取文件: {file_path}")
    print("=" * 60)
    
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return
        
        # 读取npy文件
        data = np.load(file_path, allow_pickle=True)
        
        print("✅ 文件读取成功")
        print("-" * 40)
        
        # 基本信息
        print(f"数据类型: {type(data)}")
        print(f"数据维度: {data.shape if hasattr(data, 'shape') else 'N/A'}")
        print(f"数据dtype: {data.dtype if hasattr(data, 'dtype') else 'N/A'}")
        print(f"数据大小: {data.size if hasattr(data, 'size') else 'N/A'}")
        
        # 如果是数组，显示更多信息
        if isinstance(data, np.ndarray):
            print(f"数组维度数: {data.ndim}")
            print(f"内存使用: {data.nbytes} bytes ({data.nbytes / 1024:.2f} KB)")
            
            # 显示数据范围
            if data.size > 0:
                if np.issubdtype(data.dtype, np.number):
                    print(f"最小值: {np.min(data)}")
                    print(f"最大值: {np.max(data)}")
                    print(f"平均值: {np.mean(data):.4f}")
                    print(f"标准差: {np.std(data):.4f}")
            
            # 显示数据前几个元素（如果合适的话）
            print("-" * 40)
            print("数据预览:")
            
            if data.ndim == 1:
                # 一维数组
                if data.size <= 20:
                    print(f"完整数据: {data}")
                else:
                    print(f"前10个元素: {data[:10]}")
                    print(f"后10个元素: {data[-10:]}")
            
            elif data.ndim == 2:
                # 二维数组
                print(f"形状: {data.shape}")
                if data.shape[0] <= 10 and data.shape[1] <= 10:
                    print("完整数据:")
                    print(data)
                else:
                    print("前5行、前5列:")
                    print(data[:5, :5])
            
            elif data.ndim > 2:
                # 多维数组
                print(f"多维数组形状: {data.shape}")
                print("第一个切片:")
                if data.ndim == 3:
                    print(data[0])
                else:
                    print("数据过于复杂，无法完整显示")
        
        # 如果是字典或其他对象
        elif isinstance(data, dict):
            print("字典内容:")
            for key, value in data.items():
                print(f"  {key}: {type(value)} - {value}")
        
        elif hasattr(data, '__len__'):
            print(f"序列长度: {len(data)}")
            if len(data) <= 10:
                print(f"内容: {data}")
            else:
                print(f"前5个元素: {data[:5]}")
        
        else:
            print(f"数据内容: {data}")
        
    except Exception as e:
        print(f"❌ 读取文件时出错: {str(e)}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()

def main():
    """主函数"""
    print("NPY文件信息查看器")
    print("=" * 60)
    
    # 指定要读取的npy文件路径
    npy_file_path = "D:\\pythonpro\\facial-expression-analysis-main\\models\\model_frontalization.npy"
    
    # 读取并显示信息
    read_npy_file_info(npy_file_path)
    
    print("\n" + "=" * 60)
    print("读取完成")

if __name__ == "__main__":
    main()
