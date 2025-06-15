#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看所有模型文件信息

@author: 查看脚本
@date: 2025-06-14
"""

import numpy as np
import os
from joblib import load

def read_npy_file_info(file_path):
    """读取npy文件信息"""
    print(f"📁 NPY文件: {os.path.basename(file_path)}")
    print("-" * 50)
    
    try:
        data = np.load(file_path, allow_pickle=True)
        
        print("✅ 文件读取成功")
        print(f"  数据类型: {type(data).__name__}")
        print(f"  数据维度: {data.shape if hasattr(data, 'shape') else 'N/A'}")
        print(f"  数据dtype: {data.dtype if hasattr(data, 'dtype') else 'N/A'}")
        print(f"  数据大小: {data.size if hasattr(data, 'size') else 'N/A'}")
        
        if isinstance(data, np.ndarray):
            print(f"  数组维度数: {data.ndim}")
            print(f"  内存使用: {data.nbytes} bytes ({data.nbytes / 1024:.2f} KB)")
            
            if data.size > 0 and np.issubdtype(data.dtype, np.number):
                print(f"  数值范围: [{np.min(data):.4f}, {np.max(data):.4f}]")
                print(f"  平均值: {np.mean(data):.4f}")
                print(f"  标准差: {np.std(data):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 读取失败: {str(e)}")
        return False

def read_joblib_file_info(file_path):
    """读取joblib文件信息"""
    print(f"📁 JOBLIB文件: {os.path.basename(file_path)}")
    print("-" * 50)
    
    try:
        model_data = load(file_path)
        
        print("✅ 文件读取成功")
        print(f"  数据类型: {type(model_data).__name__}")
        
        if isinstance(model_data, dict):
            print("  字典内容:")
            for key, value in model_data.items():
                if isinstance(value, np.ndarray):
                    print(f"    {key}: numpy数组 {value.shape} ({value.dtype})")
                elif hasattr(value, '__len__') and not isinstance(value, str):
                    print(f"    {key}: {type(value).__name__} (长度: {len(value)})")
                else:
                    print(f"    {key}: {value}")
        
        elif hasattr(model_data, '__dict__'):
            print("  对象属性:")
            for attr_name in dir(model_data):
                if not attr_name.startswith('_'):
                    try:
                        attr_value = getattr(model_data, attr_name)
                        if not callable(attr_value):
                            if isinstance(attr_value, np.ndarray):
                                print(f"    {attr_name}: numpy数组 {attr_value.shape}")
                            else:
                                print(f"    {attr_name}: {type(attr_value).__name__}")
                    except:
                        pass
        
        else:
            print(f"  内容: {model_data}")
        
        return True
        
    except Exception as e:
        print(f"❌ 读取失败: {str(e)}")
        return False

def read_dat_file_info(file_path):
    """读取dat文件信息"""
    print(f"📁 DAT文件: {os.path.basename(file_path)}")
    print("-" * 50)
    
    try:
        file_size = os.path.getsize(file_path)
        print("✅ 文件存在")
        print(f"  文件大小: {file_size} bytes ({file_size / 1024:.2f} KB)")
        print("  文件类型: dlib人脸特征点检测模型")
        print("  用途: 68点人脸关键点检测")
        
        return True
        
    except Exception as e:
        print(f"❌ 读取失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("🔍 模型文件信息查看器")
    print("=" * 60)
    
    models_dir = "D:\\pythonpro\\facial-expression-analysis-main\\models\\"
    
    # 定义要检查的文件
    files_to_check = [
        ("model_frontalization.npy", read_npy_file_info),
        ("model_emotion_pls=30_fullfeatures=False_py312.joblib", read_joblib_file_info),
        ("shape_predictor_68_face_landmarks.dat", read_dat_file_info)
    ]
    
    for filename, read_function in files_to_check:
        file_path = os.path.join(models_dir, filename)
        
        if os.path.exists(file_path):
            read_function(file_path)
        else:
            print(f"📁 {filename}")
            print("-" * 50)
            print(f"❌ 文件不存在: {file_path}")
        
        print("\n" + "=" * 60 + "\n")
    
    print("🎉 所有文件检查完成")

if __name__ == "__main__":
    main()
