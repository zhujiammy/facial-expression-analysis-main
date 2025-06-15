#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试升级后的模型

@author: 测试脚本
@date: 2025-06-13
"""

import sys
import os
sys.path.append('../source')


import numpy as np
import cv2
import dlib
import imageio
from joblib import load
from emotions_dlib import EmotionsDlib
import math

def test_model_loading():
    """测试模型加载"""
    print("测试模型加载...")
    
    model_path = "D:\\pythonpro\\facial-expression-analysis-main\\models\\model_emotion_pls=30_fullfeatures=False_py312.joblib"
    
    try:
        model_data = load(model_path)
        print("✅ 模型加载成功")
        print(f"  - Python 版本: {model_data.get('python_version', 'unknown')}")
        print(f"  - 组件数: {model_data['components']}")
        print(f"  - 完整特征: {model_data['full_features']}")
        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return False

def test_emotion_estimation():
    """测试情感估计"""
    print("\n测试情感估计...")
    
    try:
        # 设置检测器和模型
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("D:\\pythonpro\\facial-expression-analysis-main\\models\\shape_predictor_68_face_landmarks.dat")
        
        emotion_estimator = EmotionsDlib(
            file_emotion_model="D:\\pythonpro\\facial-expression-analysis-main\\models\\model_emotion_pls=30_fullfeatures=False_py312.joblib",
            file_frontalization_model="D:\\pythonpro\\facial-expression-analysis-main\\models\\model_frontalization.npy"
        )          # 测试图像
        test_images = [
            'D:\\pythonpro\\facial-expression-analysis-main\\data\\images\\pleased.jpg'
        ]
        
        for img_path in test_images:
            if os.path.exists(img_path):
                print(f"\n测试图像: {os.path.basename(img_path)}")
                
                image = cv2.imread(img_path)
                # 如果需要RGB格式（opencv默认是BGR）
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = detector(image)
                
                if len(faces) > 0:
                    face = faces[0]
                    landmarks_object = predictor(image, face)
                    
                    print("Landmarks object:", landmarks_object.parts())
                    
                    # 调试原始landmarks 
                    raw_landmarks = np.array([[p.x, p.y] for p in landmarks_object.parts()])
                    print(f"Raw landmarks (numpy): {raw_landmarks[:10]}")
                    
                    # 调试frontalization weights
                    print(f"Frontalization weights shape: {emotion_estimator.frontalizer.frontalization_weights.shape}")
                    print(f"First few weights: {emotion_estimator.frontalizer.frontalization_weights[0, :5]}")
                    print(f"Second row weights: {emotion_estimator.frontalizer.frontalization_weights[1, :5]}")
                    
                    # 调试正面化过程
                    dict_landmarks = emotion_estimator.frontalizer.frontalize_landmarks(landmarks_object)
                    
                    # 调试 Procrustes 标准化步骤
                    landmarks_array = np.array([[p.x, p.y] for p in landmarks_object.parts()])
                    
                    # 步骤1：平移（中心化）
                    landmark_mean = np.mean(landmarks_array, axis=0)
                    landmarks_centered = landmarks_array - landmark_mean
                    print(f"Mean landmark: {landmark_mean}")
                    print(f"First few centered landmarks: {landmarks_centered[:5]}")
                    
                    # 步骤2：缩放
                    landmark_scale = math.sqrt(np.mean(np.sum(landmarks_centered**2, axis=1)))
                    landmarks_scaled = landmarks_centered / landmark_scale
                    print(f"Calculated scale: {landmark_scale}")
                    print(f"First few scaled landmarks: {landmarks_scaled[:5]}")
                    
                    # 步骤3：旋转（调试）
                    # 计算眼睛中心
                    center_eye_left = np.mean(landmarks_scaled[36:42], axis=0)
                    center_eye_right = np.mean(landmarks_scaled[42:48], axis=0)
                    print(f"Eye centers: left{center_eye_left}, right{center_eye_right}")
                    
                    # 计算旋转角度
                    dx = center_eye_right[0] - center_eye_left[0]
                    dy = center_eye_right[1] - center_eye_left[1]
                    print(f"Eye distance: dx={dx}, dy={dy}")
                    
                    if dx != 0:
                        angle = math.atan(dy / dx)
                        print(f"Rotation angle: {angle} radians ({math.degrees(angle)} degrees)")
                    
                    # 调试 Procrustes 标准化
                    landmarks_standard = emotion_estimator.frontalizer.get_procrustes(np.array([[p.x, p.y] for p in landmarks_object.parts()]))
                    print(f"First few standardized landmarks: {landmarks_standard[:5]}")
                    
                    landmarks_frontal = dict_landmarks['landmarks_frontal']
                    landmarks_raw = dict_landmarks['landmarks_raw']
                    
                    print(f"Raw landmarks shape: {landmarks_raw.shape}")
                    print(f"Frontal landmarks shape: {landmarks_frontal.shape}")
                    print(f"First few raw landmarks: {landmarks_raw[:5]}")
                    print(f"First few frontal landmarks: {landmarks_frontal[:5]}")
                    
                    # 调试特征模板
                    print(f"Feature template shape: {emotion_estimator.geom_feat.feature_template.shape}")
                    print(f"First 10 feature pairs: {emotion_estimator.geom_feat.feature_template[:10]}")
                      # 手动计算前几个特征进行比较
                    frontal_landmarks_17_67 = landmarks_frontal[17:68]  # landmarks 17-67
                    feature_scale = emotion_estimator.geom_feat.get_scale(landmarks_frontal)
                    
                    print(f"Manual feature calculation:")
                    for i in range(min(5, len(frontal_landmarks_17_67))):
                        for j in range(i + 1, min(i + 6, len(frontal_landmarks_17_67))):
                            if j - i <= 5:  # Only first few pairs
                                p1 = frontal_landmarks_17_67[i]
                                p2 = frontal_landmarks_17_67[j]
                                raw_dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                                normalized_dist = raw_dist / feature_scale
                                actual_idx1 = i + 17
                                actual_idx2 = j + 17
                                print(f"landmarks[{actual_idx1}] ({p1[0]:.6f},{p1[1]:.6f}) and landmarks[{actual_idx2}] ({p2[0]:.6f},{p2[1]:.6f}) -> raw_dist={raw_dist:.6f}, normalized={normalized_dist:.6f}")
                    
                    # 调试特征提取过程
                    features = emotion_estimator.geom_feat.get_features(landmarks_frontal)
                    print(f"Features shape: {features.shape}")
                    print(f"First 10 features: {features[:10]}")
                    print(f"Feature min/max: {features.min():.6f} / {features.max():.6f}")
                    
                    # 调试特征提取时的scale计算
                    feature_scale = emotion_estimator.geom_feat.get_scale(landmarks_frontal)
                    print(f"Feature extraction scale: {feature_scale:.6f}")
                    
                    # 调试scale计算
                    scale = emotion_estimator.geom_feat.get_scale(landmarks_frontal)
                    print(f"Scale: {scale:.6f}")
                    
                    # 调试 landmark_indx
                    print(f"Landmark indices used: start={emotion_estimator.geom_feat.landmark_indx[0]}, end={emotion_estimator.geom_feat.landmark_indx[-1]}, count={len(emotion_estimator.geom_feat.landmark_indx)}")
                    
                    # 最终预测
                    dict_emotions = emotion_estimator.get_emotions(landmarks_object)
                    
                    arousal = dict_emotions['emotions']['arousal']
                    valence = dict_emotions['emotions']['valence']
                    intensity = dict_emotions['emotions']['intensity']
                    emotion_name = dict_emotions['emotions']['name']
                    
                    print(f"  - Arousal: {arousal}")
                    print(f"  - Valence: {valence}")
                    print(f"  - Intensity: {intensity}")
                    print(f"  - Emotion: {emotion_name}")
                else:
                    print("  - 未检测到人脸")
            else:
                print(f"  - 图像文件不存在: {img_path}")
        
        print("✅ 情感估计测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 情感估计测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("测试升级后的情感分析模型")
    print("=" * 50)
    
    # 测试模型加载
    loading_success = test_model_loading()
    
    if loading_success:
        # 测试情感估计
        estimation_success = test_emotion_estimation()
        
        if estimation_success:
            print("\n🎉 所有测试通过！模型升级成功。")
        else:
            print("\n⚠️  模型加载成功，但情感估计测试失败。")
    else:
        print("\n❌ 模型加载失败，请检查升级过程。")