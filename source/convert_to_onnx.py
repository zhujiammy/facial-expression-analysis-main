#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将joblib模型转换为ONNX格式

@author: 转换脚本
@date: 2025-06-14
"""

import numpy as np
import os
from joblib import load
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import onnxruntime as ort

def load_model_info(model_path):
    """加载并显示模型信息"""
    print(f"📁 加载模型: {os.path.basename(model_path)}")
    print("-" * 50)
    
    try:
        model_data = load(model_path)
        
        print("✅ 模型加载成功")
        print(f"  模型类型: {type(model_data['model'])}")
        print(f"  完整特征: {model_data['full_features']}")
        print(f"  组件数: {model_data['components']}")
        print(f"  Python版本: {model_data['python_version']}")
        print(f"  特征维度: {model_data['feature_dimensions']}")
        print(f"  训练样本数: {model_data['training_samples']}")
        print(f"  训练日期: {model_data['training_date']}")
        
        return model_data
        
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return None

def convert_to_onnx(model_data, output_path):
    """将sklearn模型转换为ONNX"""
    print(f"\n🔄 开始转换为ONNX格式...")
    print("-" * 50)
    
    try:
        # 获取sklearn模型
        sklearn_model = model_data['model']
        feature_dims = model_data['feature_dimensions']
        
        # 定义输入类型
        initial_type = [('float_input', FloatTensorType([None, feature_dims]))]
        
        print(f"  输入特征维度: {feature_dims}")
        print(f"  模型类型: {type(sklearn_model).__name__}")
        
        # 转换为ONNX
        onnx_model = convert_sklearn(
            sklearn_model, 
            initial_types=initial_type,
            target_opset=11  # 使用较稳定的opset版本
        )
        
        # 保存ONNX模型
        onnx.save_model(onnx_model, output_path)
        
        print(f"✅ ONNX模型已保存到: {output_path}")
        
        return onnx_model
        
    except Exception as e:
        print(f"❌ ONNX转换失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def validate_onnx_model(onnx_path, original_model_data):
    """验证ONNX模型的正确性"""
    print(f"\n🔍 验证ONNX模型...")
    print("-" * 50)
    
    try:
        # 加载ONNX模型
        ort_session = ort.InferenceSession(onnx_path)
        
        # 获取输入输出信息
        input_info = ort_session.get_inputs()[0]
        output_info = ort_session.get_outputs()
        
        print(f"  输入名称: {input_info.name}")
        print(f"  输入形状: {input_info.shape}")
        print(f"  输入类型: {input_info.type}")
        
        print(f"  输出数量: {len(output_info)}")
        for i, output in enumerate(output_info):
            print(f"  输出{i} - 名称: {output.name}, 形状: {output.shape}")
        
        # 创建测试数据
        feature_dims = original_model_data['feature_dimensions']
        test_input = np.random.randn(1, feature_dims).astype(np.float32)
        
        # ONNX模型预测
        onnx_outputs = ort_session.run(None, {input_info.name: test_input})
        
        # 原始模型预测
        sklearn_model = original_model_data['model']
        sklearn_output = sklearn_model.predict(test_input)
        
        # 比较结果
        onnx_pred = onnx_outputs[0]
        
        print(f"\n📊 预测结果比较:")
        print(f"  原始模型输出形状: {sklearn_output.shape}")
        print(f"  ONNX模型输出形状: {onnx_pred.shape}")
        
        # 计算差异
        if sklearn_output.shape == onnx_pred.shape:
            diff = np.abs(sklearn_output - onnx_pred)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"  最大差异: {max_diff:.8f}")
            print(f"  平均差异: {mean_diff:.8f}")
            
            if max_diff < 1e-5:
                print("✅ 验证通过！ONNX模型与原始模型结果一致")
                return True
            else:
                print("⚠️  存在较大差异，可能需要检查转换过程")
                return False
        else:
            print("❌ 输出形状不匹配")
            return False
            
    except Exception as e:
        print(f"❌ 验证失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def get_onnx_model_info(onnx_path):
    """获取ONNX模型详细信息"""
    print(f"\n📋 ONNX模型信息:")
    print("-" * 50)
    
    try:
        # 加载ONNX模型
        onnx_model = onnx.load(onnx_path)
        
        # 基本信息
        print(f"  ONNX版本: {onnx_model.ir_version}")
        print(f"  生产者: {onnx_model.producer_name}")
        print(f"  模型版本: {onnx_model.model_version}")
        
        # 图信息
        graph = onnx_model.graph
        print(f"  节点数量: {len(graph.node)}")
        print(f"  输入数量: {len(graph.input)}")
        print(f"  输出数量: {len(graph.output)}")
        
        # 文件大小
        file_size = os.path.getsize(onnx_path)
        print(f"  文件大小: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        
        return True
        
    except Exception as e:
        print(f"❌ 获取信息失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("🔄 Joblib到ONNX模型转换器")
    print("=" * 60)
    
    # 文件路径
    joblib_model_path = "D:\\pythonpro\\facial-expression-analysis-main\\models\\model_emotion_pls=30_fullfeatures=False_py312.joblib"
    onnx_output_path = "D:\\pythonpro\\facial-expression-analysis-main\\models\\model_emotion_pls30.onnx"
    
    # 1. 加载模型信息
    model_data = load_model_info(joblib_model_path)
    if model_data is None:
        print("❌ 无法加载原始模型，转换终止")
        return
    
    # 2. 转换为ONNX
    onnx_model = convert_to_onnx(model_data, onnx_output_path)
    if onnx_model is None:
        print("❌ ONNX转换失败，程序终止")
        return
    
    # 3. 获取ONNX模型信息
    get_onnx_model_info(onnx_output_path)
    
    # 4. 验证模型
    validation_success = validate_onnx_model(onnx_output_path, model_data)
    
    # 5. 总结
    print("\n" + "=" * 60)
    if validation_success:
        print("🎉 模型转换完成！")
        print(f"✅ ONNX模型已保存到: {onnx_output_path}")
        print("✅ 模型验证通过")
    else:
        print("⚠️  模型转换完成，但验证未通过")
        print(f"📁 ONNX模型位置: {onnx_output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()