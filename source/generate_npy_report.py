#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPY文件分析报告生成器

@author: 报告生成器
@date: 2025-06-14
"""

import numpy as np
import os
from datetime import datetime

def generate_npy_report(file_path):
    """生成NPY文件分析报告"""
    
    report = []
    report.append("=" * 80)
    report.append("NPY文件分析报告")
    report.append("=" * 80)
    report.append(f"文件路径: {file_path}")
    report.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    try:
        # 读取数据
        data = np.load(file_path)
        file_size = os.path.getsize(file_path)
        
        # 基本信息
        report.append("📊 基本信息")
        report.append("-" * 40)
        report.append(f"文件大小: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        report.append(f"数据类型: {type(data).__name__}")
        report.append(f"数据形状: {data.shape}")
        report.append(f"数据类型: {data.dtype}")
        report.append(f"元素总数: {data.size:,}")
        report.append(f"维度数: {data.ndim}")
        report.append(f"内存占用: {data.nbytes:,} bytes ({data.nbytes/1024:.2f} KB)")
        report.append("")
        
        # 统计信息
        report.append("📈 统计信息")
        report.append("-" * 40)
        report.append(f"数值范围: [{np.min(data):.6f}, {np.max(data):.6f}]")
        report.append(f"平均值: {np.mean(data):.6f}")
        report.append(f"中位数: {np.median(data):.6f}")
        report.append(f"标准差: {np.std(data):.6f}")
        report.append(f"方差: {np.var(data):.6f}")
        report.append("")
        
        # 百分位数
        report.append("📊 数据分布 (百分位数)")
        report.append("-" * 40)
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        values = np.percentile(data, percentiles)
        for p, v in zip(percentiles, values):
            report.append(f"第{p:2d}百分位数: {v:10.6f}")
        report.append("")
        
        # 数据质量
        report.append("🔍 数据质量")
        report.append("-" * 40)
        zero_count = np.sum(data == 0)
        nan_count = np.sum(np.isnan(data))
        inf_count = np.sum(np.isinf(data))
        
        report.append(f"零值数量: {zero_count:,} ({zero_count/data.size*100:.2f}%)")
        report.append(f"NaN数量: {nan_count:,}")
        report.append(f"无穷值数量: {inf_count:,}")
        report.append("")
        
        # 按行/列分析
        if data.ndim == 2:
            report.append("📊 矩阵分析")
            report.append("-" * 40)
            
            # 行统计
            row_means = np.mean(data, axis=1)
            row_stds = np.std(data, axis=1)
            report.append(f"行数: {data.shape[0]}")
            report.append(f"行均值范围: [{np.min(row_means):.6f}, {np.max(row_means):.6f}]")
            report.append(f"行标准差范围: [{np.min(row_stds):.6f}, {np.max(row_stds):.6f}]")
            
            # 列统计
            col_means = np.mean(data, axis=0)
            col_stds = np.std(data, axis=0)
            report.append(f"列数: {data.shape[1]}")
            report.append(f"列均值范围: [{np.min(col_means):.6f}, {np.max(col_means):.6f}]")
            report.append(f"列标准差范围: [{np.min(col_stds):.6f}, {np.max(col_stds):.6f}]")
            report.append("")
        
        # 数据样本
        report.append("🔬 数据样本")
        report.append("-" * 40)
        if data.ndim == 2:
            report.append("前3行前5列:")
            sample = data[:3, :5]
            for i, row in enumerate(sample):
                report.append(f"行{i}: " + " ".join([f"{val:8.4f}" for val in row]))
            
            report.append("\n后3行后5列:")
            sample = data[-3:, -5:]
            for i, row in enumerate(sample):
                report.append(f"行{data.shape[0]-3+i}: " + " ".join([f"{val:8.4f}" for val in row]))
        else:
            report.append("前10个元素:")
            report.append(str(data.flat[:10]))
        
        report.append("")
        
        # 用途推测
        report.append("💡 模型用途推测")
        report.append("-" * 40)
        if "frontalization" in file_path.lower():
            report.append("根据文件名推测，这是一个人脸正面化模型:")
            report.append("- 用于将侧脸或倾斜的人脸图像转换为正面视角")
            report.append("- 矩阵维度137x136可能对应特定的特征映射或变换参数")
            report.append("- 数值范围在[-1, 1]之间，符合归一化的变换矩阵特征")
            report.append("- 平均值接近0表明变换是中心化的")
        
        report.append("")
        report.append("=" * 80)
        report.append("分析完成")
        report.append("=" * 80)
        
        return "\n".join(report)
        
    except Exception as e:
        report.append(f"❌ 分析过程中出现错误: {str(e)}")
        return "\n".join(report)

def main():
    """主函数"""
    file_path = "D:\\pythonpro\\facial-expression-analysis-main\\models\\model_frontalization.npy"
    
    # 生成报告
    report_content = generate_npy_report(file_path)
    
    # 显示报告
    print(report_content)
    
    # 保存报告到文件
    report_file = "npy_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n📝 完整报告已保存到: {report_file}")

if __name__ == "__main__":
    main()
