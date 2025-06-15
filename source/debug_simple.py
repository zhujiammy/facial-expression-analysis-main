#!/usr/bin/env python3
"""
简化调试脚本
"""

print("开始调试...")

try:
    from emotions_dlib import EmotionsDlib
    print("✅ 成功导入EmotionsDlib")
    
    # 初始化
    emotion_analyzer = EmotionsDlib(
        file_emotion_model="../models/model_emotion_pls=30_fullfeatures=False_py312.joblib",
        file_frontalization_model="../models/model_frontalization.npy"
    )
    print("✅ EmotionsDlib初始化成功")
    
    # 加载图像并检测
    import cv2
    import dlib
    
    image = cv2.imread("../data/images/pleased.jpg")
    print(f"✅ 图像加载成功，尺寸: {image.shape}")
    
    # 检测关键点
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) > 0:
        landmarks = predictor(gray, faces[0])
        print(f"✅ 检测到人脸，关键点数量: {landmarks.num_parts}")
        
        # 处理情感
        emotions = emotion_analyzer.get_emotions(landmarks)
        
        print("=== 情感分析结果 ===")
        print(f"Arousal: {emotions['emotions']['arousal']}")
        print(f"Valence: {emotions['emotions']['valence']}")
        print(f"Intensity: {emotions['emotions']['intensity']}")
        print(f"Emotion: {emotions['emotions']['emotion_name']}")
        
    else:
        print("❌ 未检测到人脸")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("调试完成")
