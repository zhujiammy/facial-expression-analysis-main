#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug feature template generation to understand the indexing
"""

import numpy as np

class TestGeometricFeaturesDlib():
    def __init__(self, full_size=True):
        TOTAL_LANDMARKS = 68  # based on DLIB annotation
        
        if full_size is True:
            landmark_indx = tuple(i for i in range(TOTAL_LANDMARKS))   #[0,67]
        else:
            landmark_indx = tuple(i for i in range(17,TOTAL_LANDMARKS))#[17,67]
            
        # computing feature template by estimating all the unique pairs  
        # (N choose 2, where N is the number of landmarks) between all possible
        # pairs of landmaks
        
        feature_template = []  
        
        for i in range(len(landmark_indx)):
            for j in range(i+1,len(landmark_indx)):
                feature_template.append([i,j])
        
        self.feature_template = np.array(feature_template, dtype=np.int16)
        self.landmark_indx = landmark_indx
        print('Feature template size:', self.feature_template.shape)
        print('Landmark indices:', landmark_indx[:10], '...')  # first 10
        print('First 10 feature template pairs:', self.feature_template[:10])
        
        # Now let's understand what happens when we access landmarks
        if full_size:
            print("\nFor full_size=True:")
            print("landmark_indx[0] =", landmark_indx[0])  # Should be 0
            print("landmark_indx[1] =", landmark_indx[1])  # Should be 1
            print("First feature template pair [0,1] maps to landmark_indx:", 
                  (landmark_indx[0], landmark_indx[1]))
        else:
            print("\nFor full_size=False:")
            print("landmark_indx[0] =", landmark_indx[0])  # Should be 17
            print("landmark_indx[1] =", landmark_indx[1])  # Should be 18
            print("First feature template pair [0,1] maps to landmark_indx:", 
                  (landmark_indx[0], landmark_indx[1]))
            
    def test_feature_access(self, landmarks_dlib):
        """Test how features are actually accessed"""
        print("\nTesting feature access...")
        print("Feature template[0]:", self.feature_template[0])  # Should be [0,1]
        
        # The way features are accessed in get_features:
        idx0 = self.feature_template[0,0]  # This is 0 (relative index)
        idx1 = self.feature_template[0,1]  # This is 1 (relative index)
        
        print("Template indices for first feature:", idx0, idx1)
        
        # These are used to index into the landmarks array:
        point0 = landmarks_dlib[idx0]
        point1 = landmarks_dlib[idx1]
        
        print("landmarks_dlib[{}]: {}".format(idx0, point0))
        print("landmarks_dlib[{}]: {}".format(idx1, point1))
        
        # BUT - this assumes that landmarks_dlib is the FULL 68-landmark array
        # and the feature template indices are absolute indices into that array
        
        return point0, point1

if __name__ == "__main__":
    print("Testing full_size=True:")
    geom_full = TestGeometricFeaturesDlib(full_size=True)
    
    print("\n" + "="*50)
    print("Testing full_size=False:")
    geom_partial = TestGeometricFeaturesDlib(full_size=False)
    
    # Create dummy landmarks
    dummy_landmarks = np.random.rand(68, 2)
    
    print("\n" + "="*50)
    print("Testing feature access with full_size=False:")
    geom_partial.test_feature_access(dummy_landmarks)
