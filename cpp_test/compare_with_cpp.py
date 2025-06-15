#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
与C++模型比较的Python脚本

@author: 比较脚本
@date: 2025-06-14
"""

import sys
import os
import numpy as np
import cv2
import dlib
from joblib import load

# 添加源代码路径
sys.path.append('../source')
from emotions_dlib import EmotionsDlib

def load_models():
    """加载模型"""
    print("加载Python模型...")
    
    # 模型路径
    onnx_model_path = "../models/model_emotion_pls30.onnx"
    joblib_model_path = "../models/model_emotion_pls=30_fullfeatures=False_py312.joblib"
    frontalization_model_path = "../models/model_frontalization.npy"
    shape_predictor_path = "../models/shape_predictor_68_face_landmarks.dat"
    
    # 加载joblib模型进行比较
    model_data = load(joblib_model_path)
    sklearn_model = model_data['model']
    
    # 加载情感分析器
    emotion_estimator = EmotionsDlib(
        file_emotion_model=joblib_model_path,
        file_frontalization_model=frontalization_model_path
    )
    
    # 加载dlib检测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)
    
    return emotion_estimator, detector, predictor, sklearn_model

def analyze_image_with_python(image_path, emotion_estimator, detector, predictor):
    """使用Python模型分析图像"""
    print(f"分析图像: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        return None
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    # 转换为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 检测面部
    faces = detector(image)
    if len(faces) == 0:
        print("未检测到面部")
        return None
    
    # 使用第一个检测到的面部
    face = faces[0]
    landmarks_object = predictor(image, face)
    
    # 进行情感分析
    dict_emotions = emotion_estimator.get_emotions(landmarks_object)
    
    arousal = dict_emotions['emotions']['arousal']
    valence = dict_emotions['emotions']['valence']
    intensity = dict_emotions['emotions']['intensity']
    emotion_name = dict_emotions['emotions']['name']
    
    return {
        'arousal': arousal,
        'valence': valence,
        'intensity': intensity,
        'emotion_name': emotion_name,
        'landmarks_raw': dict_emotions['landmarks']['raw'],
        'landmarks_frontal': dict_emotions['landmarks']['frontal']
    }

def test_random_features(sklearn_model, num_samples=10, feature_dims=1275):
    """测试随机特征向量"""
    print(f"生成 {num_samples} 个随机特征向量进行测试...")
    
    # 加载测试特征（应该由C++程序生成）
    features_file = "test_features.txt"
    if not os.path.exists(features_file):
        print(f"特征文件不存在: {features_file}")
        return []
    
    # 读取特征
    features = []
    with open(features_file, 'r') as f:
        for line in f:
            feature_row = [float(x) for x in line.strip().split(',')]
            features.append(feature_row)
    
    print(f"读取到 {len(features)} 个特征向量")
    
    # 使用sklearn模型预测
    predictions = []
    for feature in features:
        feature_array = np.array(feature).reshape(1, -1)
        pred = sklearn_model.predict(feature_array)
        predictions.append(pred[0].tolist())
    
    return predictions

def save_python_predictions(predictions, output_file):
    """保存Python预测结果"""
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(','.join([str(x) for x in pred]) + '\n')
    print(f"Python预测结果已保存到: {output_file}")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python compare_with_cpp.py <mode> [args...]")
        print("模式:")
        print("  images <image1> [image2] ...  - 比较图像预测")
        print("  random                        - 比较随机特征预测")
        return
    
    mode = sys.argv[1]
    
    try:
        # 加载模型
        emotion_estimator, detector, predictor, sklearn_model = load_models()
        
        if mode == "images":
            # 图像模式
            if len(sys.argv) < 3:
                print("图像模式需要指定至少一个图像文件")
                return
            
            image_paths = sys.argv[2:]
            predictions = []
            
            for image_path in image_paths:
                result = analyze_image_with_python(image_path, emotion_estimator, detector, predictor)
                if result:
                    predictions.append([result['arousal'], result['valence']])
                    print(f"  Python结果 - Arousal: {result['arousal']:.6f}, Valence: {result['valence']:.6f}")
                else:
                    predictions.append([0.0, 0.0])  # 默认值
            
            # 保存结果
            save_python_predictions(predictions, "python_predictions.txt")
            
        elif mode == "random":
            # 随机特征模式
            predictions = test_random_features(sklearn_model)
            if predictions:
                save_python_predictions(predictions, "python_random_predictions.txt")
            else:
                print("随机特征测试失败")
        
        else:
            print(f"未知模式: {mode}")
            return
        
        print("Python比较脚本执行完成")
        
    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
