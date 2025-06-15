#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深入分析NPY文件内容

@author: 分析脚本
@date: 2025-06-14
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_frontalization_model(file_path):
    """深入分析frontalization模型"""
    print("🔬 深入分析 model_frontalization.npy")
    print("=" * 60)
    
    try:
        data = np.load(file_path)
        
        print("📊 基本信息:")
        print(f"  形状: {data.shape}")
        print(f"  数据类型: {data.dtype}")
        print(f"  内存大小: {data.nbytes / 1024:.2f} KB")
        
        print("\n📈 统计信息:")
        print(f"  最小值: {np.min(data):.6f}")
        print(f"  最大值: {np.max(data):.6f}")
        print(f"  平均值: {np.mean(data):.6f}")
        print(f"  中位数: {np.median(data):.6f}")
        print(f"  标准差: {np.std(data):.6f}")
        print(f"  方差: {np.var(data):.6f}")
        
        # 分析每行的统计信息
        print("\n📊 每行统计信息:")
        row_means = np.mean(data, axis=1)
        row_stds = np.std(data, axis=1)
        
        print(f"  行均值范围: [{np.min(row_means):.6f}, {np.max(row_means):.6f}]")
        print(f"  行标准差范围: [{np.min(row_stds):.6f}, {np.max(row_stds):.6f}]")
        
        # 分析每列的统计信息
        print("\n📊 每列统计信息:")
        col_means = np.mean(data, axis=0)
        col_stds = np.std(data, axis=0)
        
        print(f"  列均值范围: [{np.min(col_means):.6f}, {np.max(col_means):.6f}]")
        print(f"  列标准差范围: [{np.min(col_stds):.6f}, {np.max(col_stds):.6f}]")
        
        # 查找零值和异常值
        print("\n🔍 数据质量分析:")
        zero_count = np.sum(data == 0)
        nan_count = np.sum(np.isnan(data))
        inf_count = np.sum(np.isinf(data))
        
        print(f"  零值数量: {zero_count} ({zero_count/data.size*100:.2f}%)")
        print(f"  NaN数量: {nan_count}")
        print(f"  无穷值数量: {inf_count}")
        
        # 分析数据分布
        print("\n📊 数据分布分析:")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        values = np.percentile(data, percentiles)
        
        for p, v in zip(percentiles, values):
            print(f"  第{p}百分位数: {v:.6f}")
        
        # 显示数据的一些样本
        print("\n🔬 数据样本:")
        print("前3行前5列:")
        print(data[:3, :5])
        
        print("\n最后3行最后5列:")
        print(data[-3:, -5:])
        
        # 创建可视化
        create_visualizations(data)
        
        return True
        
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_visualizations(data):
    """创建数据可视化"""
    print("\n📊 创建可视化图表...")
    
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Frontalization模型数据分析', fontsize=16)
        
        # 1. 热力图 (采样显示，因为数据太大)
        sample_data = data[::5, ::5]  # 每5个取一个点
        im1 = axes[0, 0].imshow(sample_data, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('数据热力图 (采样)')
        axes[0, 0].set_xlabel('列索引')
        axes[0, 0].set_ylabel('行索引')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. 数据分布直方图
        axes[0, 1].hist(data.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('数据值分布')
        axes[0, 1].set_xlabel('数值')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 每行的均值
        row_means = np.mean(data, axis=1)
        axes[1, 0].plot(row_means, 'b-', linewidth=1)
        axes[1, 0].set_title('每行均值')
        axes[1, 0].set_xlabel('行索引')
        axes[1, 0].set_ylabel('均值')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 每列的均值
        col_means = np.mean(data, axis=0)
        axes[1, 1].plot(col_means, 'r-', linewidth=1)
        axes[1, 1].set_title('每列均值')
        axes[1, 1].set_xlabel('列索引')
        axes[1, 1].set_ylabel('均值')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = "frontalization_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ 可视化图表已保存到: {output_path}")
        
        # 显示图表
        plt.show()
        
    except Exception as e:
        print(f"  ❌ 可视化创建失败: {str(e)}")

def main():
    """主函数"""
    print("🔬 NPY文件深度分析器")
    print("=" * 60)
    
    npy_file_path = "D:\\pythonpro\\facial-expression-analysis-main\\models\\model_frontalization.npy"
    
    if os.path.exists(npy_file_path):
        analyze_frontalization_model(npy_file_path)
    else:
        print(f"❌ 文件不存在: {npy_file_path}")
    
    print("\n" + "=" * 60)
    print("🎉 分析完成")

if __name__ == "__main__":
    main()
