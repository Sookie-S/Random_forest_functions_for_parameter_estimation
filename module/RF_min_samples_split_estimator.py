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

def estimate_min_samples_split(X, y, testSize, outdir):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)

    #########
    # Estimation of the minimum number of samples required to split an internal node
    #########

    # min_samples_splits = list(range(2, 1001, 5))  # Initial Default range for min_samples_split
    min_samples_splits = list(range(2, 451))  # Default range for min_samples_split
    train_results = []
    test_results = []

    for min_samples_split in min_samples_splits:
        rf = RandomForestClassifier(min_samples_split=min_samples_split, random_state=42)
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

    # Find optimal min_samples_split
    best_idx = np.argmax(test_results)
    best_min_samples_split = min_samples_splits[best_idx]  # Map index to parameter value

    # Plot AUC vs min_samples_split
    paramDir = os.path.join(outdir, "Parameters_estimation")
    os.makedirs(paramDir, exist_ok=True)

    plt.plot(min_samples_splits, train_results, 'b', label="Train AUC")
    plt.plot(min_samples_splits, test_results, 'r', label="Test AUC")
    plt.legend()
    plt.ylabel('AUC Score')
    plt.xlabel('Min Samples Split')
    plt.title('Random Forest AUC vs Min Samples Split')
    plt.savefig(os.path.join(paramDir, 'RF_auc_per_min_sample_split.png'))
    plt.close()

    # Return optimal parameter
    return best_min_samples_split
