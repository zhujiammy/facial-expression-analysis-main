#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成用于C++程序测试的Python参考结果

@author: 测试脚本
@date: 2025-06-14
"""

import sys
import os
import numpy as np
import json
from pathlib import Path

# 添加源码路径
sys.path.append('../source')

try:
    from joblib import load
    from emotions_dlib import EmotionsDlib
    import cv2
    import dlib
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保安装了所有必要的依赖项")
    sys.exit(1)

class PythonReferenceGenerator:
    """生成Python参考结果的类"""
    
    def __init__(self, models_dir="../models"):
        self.models_dir = Path(models_dir)
        self.joblib_model_path = self.models_dir / "model_emotion_pls=30_fullfeatures=False_py312.joblib"
        self.frontalization_model_path = self.models_dir / "model_frontalization.npy"
        self.shape_predictor_path = self.models_dir / "shape_predictor_68_face_landmarks.dat"
        
        self.emotion_estimator = None
        self.detector = None
        self.predictor = None
        self.sklearn_model = None
        
    def load_models(self):
        """加载所有必要的模型"""
        print("加载Python模型...")
        
        # 检查文件是否存在
        for path in [self.joblib_model_path, self.frontalization_model_path, self.shape_predictor_path]:
            if not path.exists():
                raise FileNotFoundError(f"模型文件不存在: {path}")
        
        # 加载joblib模型
        model_data = load(str(self.joblib_model_path))
        self.sklearn_model = model_data['model']
        
        print(f"模型组件数: {model_data['components']}")
        print(f"完整特征: {model_data['full_features']}")
        
        # 加载情感分析器
        self.emotion_estimator = EmotionsDlib(
            file_emotion_model=str(self.joblib_model_path),
            file_frontalization_model=str(self.frontalization_model_path)
        )
        
        # 加载dlib检测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(self.shape_predictor_path))
        
        print("✅ 所有模型加载成功")
        
    def analyze_image(self, image_path):
        """分析单张图像"""
        print(f"分析图像: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 检测面部
        faces = self.detector(image)
        if len(faces) == 0:
            raise ValueError("未检测到面部")
        
        # 使用第一个检测到的面部
        face = faces[0]
        landmarks_object = self.predictor(image, face)
        
        # 进行情感分析
        dict_emotions = self.emotion_estimator.get_emotions(landmarks_object)
        
        result = {
            'arousal': float(dict_emotions['emotions']['arousal']),
            'valence': float(dict_emotions['emotions']['valence']),
            'intensity': float(dict_emotions['emotions']['intensity']),
            'emotion_name': dict_emotions['emotions']['name'],
            'landmarks_raw': [[float(p.x), float(p.y)] for p in landmarks_object.parts()],
            'landmarks_frontal': [[float(p[0]), float(p[1])] for p in dict_emotions['landmarks']['frontal']]
        }
        
        return result
    
    def test_random_features(self, features_file="test_features.txt"):
        """测试随机特征向量"""
        print(f"测试随机特征向量: {features_file}")
        
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"特征文件不存在: {features_file}")
        
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
            pred = self.sklearn_model.predict(feature_array)
            predictions.append(pred[0].tolist())
        
        return predictions
    
    def generate_synthetic_test_data(self, num_samples=10):
        """生成合成测试数据"""
        print(f"生成 {num_samples} 个合成测试样本...")
        
        # 生成随机特征（基于训练数据的统计特性）
        np.random.seed(42)  # 确保可重现
        
        # 假设特征维度为1275（不包含下颚线的特征）
        feature_dim = 1275
        
        # 生成随机特征
        features = np.random.randn(num_samples, feature_dim).astype(np.float32)
        
        # 使用模型预测
        predictions = []
        for i in range(num_samples):
            pred = self.sklearn_model.predict(features[i:i+1])
            predictions.append(pred[0].tolist())
        
        # 保存测试数据
        np.savetxt("synthetic_features.txt", features, delimiter=',', fmt='%.6f')
        
        # 保存Python预测结果
        with open("python_synthetic_predictions.txt", 'w') as f:
            for pred in predictions:
                f.write(','.join([f'{x:.6f}' for x in pred]) + '\n')
        
        print("✅ 合成测试数据生成完成")
        return features, predictions
    
    def save_results(self, results, output_file):
        """保存结果到文件"""
        if output_file.endswith('.json'):
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            # 保存为CSV格式
            with open(output_file, 'w') as f:
                for result in results:
                    if isinstance(result, dict):
                        f.write(f"{result['arousal']:.6f},{result['valence']:.6f}\n")
                    else:
                        f.write(','.join([f'{x:.6f}' for x in result]) + '\n')
        
        print(f"结果已保存到: {output_file}")

def main():
    """主函数"""
    print("========================================")
    print("   Python参考结果生成器")
    print("========================================")
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python generate_reference.py image <image_path1> [image_path2] ...")
        print("  python generate_reference.py random <features_file>")
        print("  python generate_reference.py synthetic [num_samples]")
        return
    
    mode = sys.argv[1]
    
    try:
        generator = PythonReferenceGenerator()
        generator.load_models()
        
        if mode == "image":
            # 图像分析模式
            if len(sys.argv) < 3:
                print("错误: 图像模式需要指定至少一个图像文件")
                return
            
            image_paths = sys.argv[2:]
            results = []
            
            for image_path in image_paths:
                try:
                    result = generator.analyze_image(image_path)
                    results.append(result)
                    print(f"✅ {image_path}: arousal={result['arousal']:.6f}, valence={result['valence']:.6f}")
                except Exception as e:
                    print(f"❌ {image_path}: {str(e)}")
                    results.append({'arousal': 0.0, 'valence': 0.0, 'error': str(e)})
            
            # 保存结果
            generator.save_results(results, "python_image_results.json")
            
            # 也保存为C++可读的格式
            simple_results = [[r['arousal'], r['valence']] for r in results]
            generator.save_results(simple_results, "python_predictions.txt")
            
        elif mode == "random":
            # 随机特征测试模式
            features_file = sys.argv[2] if len(sys.argv) > 2 else "test_features.txt"
            predictions = generator.test_random_features(features_file)
            generator.save_results(predictions, "python_random_predictions.txt")
            
        elif mode == "synthetic":
            # 合成数据模式
            num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            features, predictions = generator.generate_synthetic_test_data(num_samples)
            
        else:
            print(f"未知模式: {mode}")
            return
        
        print("\n🎉 Python参考结果生成完成！")
        
    except Exception as e:
        print(f"❌ 执行失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
