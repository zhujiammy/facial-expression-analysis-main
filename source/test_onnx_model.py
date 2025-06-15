#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试ONNX模型

@author: 测试脚本
@date: 2025-06-14
"""

import numpy as np
import onnxruntime as ort
from joblib import load
import os

def test_onnx_model():
    """测试ONNX模型的功能"""
    print("🧪 测试ONNX模型")
    print("=" * 50)
    
    # 文件路径
    onnx_model_path = "D:\\pythonpro\\facial-expression-analysis-main\\models\\model_emotion_pls30.onnx"
    joblib_model_path = "D:\\pythonpro\\facial-expression-analysis-main\\models\\model_emotion_pls=30_fullfeatures=False_py312.joblib"
    
    try:
        # 1. 加载ONNX模型
        print("📁 加载ONNX模型...")
        ort_session = ort.InferenceSession(onnx_model_path)
        
        # 获取输入输出信息
        input_info = ort_session.get_inputs()[0]
        output_info = ort_session.get_outputs()
        
        print(f"✅ ONNX模型加载成功")
        print(f"  输入名称: {input_info.name}")
        print(f"  输入形状: {input_info.shape}")
        print(f"  输出数量: {len(output_info)}")
        
        # 2. 加载原始joblib模型
        print("\n📁 加载原始joblib模型...")
        model_data = load(joblib_model_path)
        sklearn_model = model_data['model']
        feature_dims = model_data['feature_dimensions']
        
        print(f"✅ Joblib模型加载成功")
        print(f"  特征维度: {feature_dims}")
        
        # 3. 创建测试数据
        print(f"\n🔬 创建测试数据...")
        np.random.seed(42)  # 固定随机种子以便重现
        test_samples = 5
        test_input = np.random.randn(test_samples, feature_dims).astype(np.float32)
        
        print(f"  测试样本数: {test_samples}")
        print(f"  输入数据形状: {test_input.shape}")
        
        # 4. 进行预测比较
        print(f"\n📊 预测结果比较:")
        
        # 原始模型预测
        sklearn_predictions = sklearn_model.predict(test_input)
        
        # ONNX模型预测
        onnx_predictions = ort_session.run(None, {input_info.name: test_input})[0]
        
        print(f"  原始模型输出形状: {sklearn_predictions.shape}")
        print(f"  ONNX模型输出形状: {onnx_predictions.shape}")
        
        # 计算差异
        diff = np.abs(sklearn_predictions - onnx_predictions)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"  最大差异: {max_diff:.8f}")
        print(f"  平均差异: {mean_diff:.8f}")
        
        # 显示具体预测结果
        print(f"\n📋 详细预测结果对比:")
        for i in range(min(3, test_samples)):  # 只显示前3个样本
            print(f"  样本 {i+1}:")
            print(f"    原始模型: {sklearn_predictions[i]}")
            print(f"    ONNX模型: {onnx_predictions[i]}")
            print(f"    差异: {diff[i]}")
        
        # 5. 性能测试
        print(f"\n⚡ 性能测试:")
        import time
        
        # 测试原始模型速度
        start_time = time.time()
        for _ in range(100):
            _ = sklearn_model.predict(test_input[:1])
        sklearn_time = time.time() - start_time
        
        # 测试ONNX模型速度
        start_time = time.time()
        for _ in range(100):
            _ = ort_session.run(None, {input_info.name: test_input[:1]})
        onnx_time = time.time() - start_time
        
        print(f"  原始模型 (100次预测): {sklearn_time:.4f}秒")
        print(f"  ONNX模型 (100次预测): {onnx_time:.4f}秒")
        print(f"  速度比较: ONNX是原始模型的 {sklearn_time/onnx_time:.2f}倍快")
        
        # 6. 验证结果
        print(f"\n✅ 测试结果:")
        if max_diff < 1e-5:
            print("  🎉 模型转换成功！预测结果一致")
            print(f"  📁 ONNX模型文件: {onnx_model_path}")
            file_size = os.path.getsize(onnx_model_path)
            print(f"  📊 文件大小: {file_size:,} bytes ({file_size/1024:.2f} KB)")
            return True
        else:
            print("  ⚠️  存在较大预测差异，请检查转换过程")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = test_onnx_model()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ONNX模型测试通过！")
        print("✅ 模型已成功转换并可以正常使用")
    else:
        print("❌ ONNX模型测试失败")
    print("=" * 50)

if __name__ == "__main__":
    main()
