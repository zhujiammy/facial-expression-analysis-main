#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试Python和C++差异
"""

import numpy as np
from emotions_dlib import EmotionsDlib
import cv2
import dlib

def test_pleased_image():    # 初始化
    emotion_analyzer = EmotionsDlib(
        file_emotion_model="../models/model_emotion_pls=30_fullfeatures=False_py312.joblib",
        file_frontalization_model="../models/model_frontalization.npy"
    )
    
    # 加载图像
    image_path = "../data/images/pleased.jpg"
    image = cv2.imread(image_path)
    
    # 检测关键点
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) > 0:
        landmarks = predictor(gray, faces[0])
        
        print("=== 原始关键点 (前10个) ===")
        for i in range(10):
            point = landmarks.part(i)
            print(f"  Landmark {i}: ({point.x}, {point.y})")
        
        # 使用emotion_analyzer处理
        emotions = emotion_analyzer.get_emotions(landmarks)
        
        print("\n=== 情感分析结果 ===")
        print(f"Arousal: {emotions['emotions']['arousal']}")
        print(f"Valence: {emotions['emotions']['valence']}")
        print(f"Intensity: {emotions['emotions']['intensity']}")
        print(f"Emotion: {emotions['emotions']['emotion_name']}")
        
        # 获取frontalization过程的详细信息
        print("\n=== 正面化过程调试 ===")
        
        # 手动调用frontalization
        dict_landmarks = emotion_analyzer.frontalizer.frontalize_landmarks(landmarks)
        landmarks_frontal = dict_landmarks['landmarks_frontal']
        landmarks_raw = dict_landmarks['landmarks_raw']
        
        print(f"Raw landmarks shape: {landmarks_raw.shape}")
        print(f"Frontal landmarks shape: {landmarks_frontal.shape}")
        
        print("\n前3个原始关键点:")
        for i in range(3):
            print(f"  Raw {i}: ({landmarks_raw[i,0]:.6f}, {landmarks_raw[i,1]:.6f})")
            
        print("\n前3个正面化关键点:")
        for i in range(3):
            print(f"  Frontal {i}: ({landmarks_frontal[i,0]:.6f}, {landmarks_frontal[i,1]:.6f})")
        
        # 获取特征
        features = emotion_analyzer.geom_feat.get_features(landmarks_frontal)
        print(f"\n特征向量大小: {features.shape}")
        print(f"前5个特征值: {features[:5]}")
        
        # 获取scale信息
        scale = emotion_analyzer.geom_feat.get_scale(landmarks_frontal)
        print(f"Scale值: {scale}")

if __name__ == "__main__":
    test_pleased_image()
