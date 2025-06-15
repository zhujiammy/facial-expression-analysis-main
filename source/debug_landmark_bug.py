#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test to verify the landmark indexing bug
"""

import sys
import os
sys.path.append('../source')
import numpy as np
from emotions_dlib import GeometricFeaturesDlib

# Create test data
dummy_landmarks = np.zeros((68, 2))
for i in range(68):
    dummy_landmarks[i] = [i * 10, i * 10 + 5]  # Make it easy to track

print("Test landmarks:")
for i in range(68):
    print(f"Landmark {i}: ({dummy_landmarks[i,0]}, {dummy_landmarks[i,1]})")

# Test with full_size=False
geom_feat = GeometricFeaturesDlib(full_size=False)

print(f"\nGeometric features landmark_indx: {geom_feat.landmark_indx}")
print(f"Feature template shape: {geom_feat.feature_template.shape}")
print(f"First few feature template pairs: {geom_feat.feature_template[:5]}")

# Test scale calculation
scale = geom_feat.get_scale(dummy_landmarks)
print(f"\nScale calculation uses landmarks: {geom_feat.landmark_indx}")

# Let's manually check what landmarks are used in get_scale
scale_landmarks = dummy_landmarks[geom_feat.landmark_indx, :]
print(f"First few landmarks used in scale calculation:")
for i in range(5):
    actual_idx = geom_feat.landmark_indx[i]
    print(f"  landmark_indx[{i}] = {actual_idx} -> ({scale_landmarks[i,0]}, {scale_landmarks[i,1]})")

# Test feature extraction - let's see which landmarks are actually used
print(f"\nFeature extraction:")
print(f"First feature template pair: {geom_feat.feature_template[0]}")
idx0, idx1 = geom_feat.feature_template[0]
print(f"This uses landmarks[{idx0}] and landmarks[{idx1}]")
print(f"  landmarks[{idx0}]: ({dummy_landmarks[idx0,0]}, {dummy_landmarks[idx0,1]})")
print(f"  landmarks[{idx1}]: ({dummy_landmarks[idx1,0]}, {dummy_landmarks[idx1,1]})")

print(f"\nExpected to use landmarks 17-67, but actually using landmarks 0-50!")
print(f"Landmark 17 should be: ({dummy_landmarks[17,0]}, {dummy_landmarks[17,1]})")
print(f"But feature extraction uses landmark 0: ({dummy_landmarks[0,0]}, {dummy_landmarks[0,1]})")
