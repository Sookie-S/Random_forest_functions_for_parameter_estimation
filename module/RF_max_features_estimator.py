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

#%%
#############################################################################################################

#### Random forest functions

#############################################################################################################

def estimate_max_features(X, y, testSize, outdir):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)
    
    #########
    # Estimation of the minimum number of features to consider when looking for the best split
    #########
    
    p = X_train.shape[1]  # Number of features
    # max_features_values = [
    #     int(np.sqrt(p)),       # sqrt(p)
    #     int(np.log2(p)),       # log2(p)
    #     int(p / 10),           # p/10
    #     int(p / 3),            # p/3
    # ]
    max_features_values = list(range(1, int(p / 10) + 1))
    
    train_results = []
    test_results = []

    for max_feature in max_features_values:
        rf = RandomForestClassifier(max_features=max_feature, random_state=42)
        rf.fit(X_train, y_train)
        
        # Training set performance
        train_pred = rf.predict(X_train)
        false_positive_rate, true_positive_rate, _ = roc_curve(y_train, train_pred)
        train_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(train_auc)
        
        # Testing set performance
        test_pred = rf.predict(X_test)
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, test_pred)
        test_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(test_auc)

    # Find the optimal max_features based on test AUC
    best_max_feature_idx = np.argmax(test_results)  # Get the index of the highest AUC
    best_max_feature = max_features_values[best_max_feature_idx]  # Map index to max_features value
    
    # Calculate sqrt of features for diversity
    features_sqrt_opt = int(np.sqrt(p))

    # Plot max_features vs AUC
    paramDir = os.path.join(outdir, "Parameters_estimation")
    os.makedirs(paramDir, exist_ok=True)

    plt.plot(max_features_values, train_results, 'b', label="Train AUC")
    plt.plot(max_features_values, test_results, 'r', label="Test AUC")
    plt.legend()
    plt.ylabel('AUC score')
    plt.xlabel('Max Features')
    plt.title('Random Forest AUC vs Max Features')
    plt.savefig(os.path.join(paramDir, 'RF_auc_per_max_features.png'))
    plt.close()

    # Return optimal parameters
    return best_max_feature, features_sqrt_opt
