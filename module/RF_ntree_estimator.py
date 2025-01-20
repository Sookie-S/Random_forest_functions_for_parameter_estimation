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
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

#%%
#############################################################################################################

#### Random forest functions

#############################################################################################################

def estimate_ntree(X, y, testSize, outdir):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)

    #########
    # Estimation of the optimal number of trees
    #########

    n_trees = list(range(1, 1501))
    accuracy_scores = []
    train_results = []
    test_results = []

    for n in n_trees:
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        rf.fit(X_train, y_train)

        # Predict once for test data
        y_pred_test = rf.predict(X_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred_test))

        # Training AUC
        train_pred = rf.predict(X_train)
        fpr_train, tpr_train, _ = roc_curve(y_train, train_pred)
        train_auc = auc(fpr_train, tpr_train)
        train_results.append(train_auc)

        # Testing AUC
        fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_test)
        test_auc = auc(fpr_test, tpr_test)
        test_results.append(test_auc)

    # Optimal number of trees based on accuracy and AUC
    ntree_opt_acc = n_trees[np.argmax(accuracy_scores)]
    ntree_opt_auc = n_trees[np.argmax(test_results)]

    # Plot Accuracy vs Number of Trees
    paramDir = os.path.join(outdir, "Parameters_estimation")
    os.makedirs(paramDir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(n_trees, accuracy_scores, marker='o', label="Accuracy")
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title("Random Forest Accuracy vs Number of Trees")
    plt.savefig(os.path.join(paramDir, 'RF_accuracy_per_ntree.png'))
    plt.close()

    # Plot AUC vs Number of Trees
    plt.figure(figsize=(8, 6))
    plt.plot(n_trees, train_results, 'b', label="Train AUC")
    plt.plot(n_trees, test_results, 'r', label="Test AUC")
    plt.legend()
    plt.xlabel('Number of Trees')
    plt.ylabel('AUC Score')
    plt.title("AUC Scores vs Number of Trees")
    plt.savefig(os.path.join(paramDir, 'RF_AUC_score_per_ntree.png'))
    plt.close()

    # Return optimal parameters
    return ntree_opt_auc, ntree_opt_acc

