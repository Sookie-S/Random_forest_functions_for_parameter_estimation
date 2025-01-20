#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Soukaina Timouma
@email: soukaina.timouma@well.OX.AC.UK
"""

#%%
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import math
#%%
#############################################################################################################

#### Random forest functions

#############################################################################################################

def estimate_max_depth(X, y, testSize, outdir):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)  # 42 is a reference from Hitchhikers guide to galaxy book. The answer to life universe and everything and is meant as a joke

    #########
    # Estimation of optimal depth of trees
    #########

    # Baseline and adjustment for tree depth
    log2_features = int(math.log2(X_train.shape[1]))  # log2​(n_features) gives a baseline estimate of how many splits are needed to explore a meaningful number of feature combinations
    # k=5 # Adding k allows the model to explore deeper relationships in the data without being overly restricted by log⁡2(n_features)
    # with our data, k small (1-3) can lead to under fitting, while k large (10-20) can lead to over fitting, k=5 works well is a reasonable starting point
    # small features (<100) k between 1 and 3 to avoid overly deep trees
    # large features (>10000) k between 3 and 10 to avoid overly deep trees, we can consider k when extra large and complex dataset (millions of features)
    
    # max_depths = list(range(1, log2_features + k + 5))  # Initial Depth range to explore
    max_depths = list(range(1, log2_features))  # Depth range to explore

    train_results = []
    test_results = []

    for max_depth in max_depths:
        rf = RandomForestClassifier(max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)

        # Training set AUC
        train_pred = rf.predict(X_train)
        false_positive_rate, true_positive_rate, _ = roc_curve(y_train, train_pred)
        train_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(train_auc)

        # Testing set AUC
        test_pred = rf.predict(X_test)
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, test_pred)
        test_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(test_auc)

    # Optimal depth and adjusted depth
    best_depth_idx = np.argmax(test_results)
    best_max_depth = max_depths[best_depth_idx]  # Map index to depth value
    # depth_log2_opt = max(10, log2_features + k)  # Ensure minimum reasonable depth
    depth_log2_opt = max(10, log2_features)  # Ensure minimum reasonable depth

    # Plot AUC vs Tree Depth
    paramDir = os.path.join(outdir, "Parameters_estimation")
    os.makedirs(paramDir, exist_ok=True)

    plt.plot(max_depths, train_results, 'b', label='Train AUC')
    plt.plot(max_depths, test_results, 'r', label='Test AUC')
    plt.legend()
    plt.ylabel('AUC Score')
    plt.xlabel('Tree Depth')
    plt.title('Random Forest AUC vs Tree Depth')
    plt.savefig(os.path.join(paramDir, 'RF_auc_per_tree_depth.png'))
    plt.close()

    # Return parameters
    return best_max_depth, depth_log2_opt
