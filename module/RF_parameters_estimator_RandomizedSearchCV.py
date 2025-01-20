#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Soukaina Timouma
@email: soukaina.timouma@well.OX.AC.UK
"""

#%%
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import math
#%%
#############################################################################################################

#### Random forest functions

#############################################################################################################

def runRandomizedSearchCV(X, y, testSize, outdir, n_iter=50, scoring='accuracy'):
    """
    Perform RandomizedSearchCV to tune hyperparameters of RandomForestClassifier.

    Parameters:
        X (array-like): Features.
        y (array-like): Target labels.
        testSize (float): Fraction of data to use for testing.
        outdir (str): Output directory for results.
        n_iter (int): Number of parameter combinations to try. Default is 50.
        scoring (str): Scoring metric to optimize. Default is 'accuracy'.

    Returns:
        tuple: Optimal parameters for RandomForestClassifier.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)

    # Define the parameter distributions
    log2_features = int(math.log2(X_train.shape[1]))
    p = X_train.shape[1]
    param_dist = {
        'n_estimators': list(range(1, 1501)),
        'max_depth': list(range(1, log2_features)),
        'min_samples_split': list(range(2, 451)),
        'min_samples_leaf': list(range(1, 61)),
        'max_features': list(range(1, int(p / 10) + 1)),
        'bootstrap': [True, False]
    }

    # Create the model
    rf = RandomForestClassifier(random_state=42)

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=5,  # 5-fold cross-validation
        random_state=42,  # Reproducibility
        n_jobs=-1,  # Use all available processors
        verbose=2  # Display progress
    )

    # Fit RandomizedSearchCV on training data
    random_search.fit(X_train, y_train)

    # Best parameters and score
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("Best Parameters:", best_params)
    print(f"Best Cross-Validation {scoring.capitalize()}:", best_score)

    # Prepare output directory
    paramDir = os.path.join(outdir, "Parameters_estimation")
    os.makedirs(paramDir, exist_ok=True)

    # Save detailed results
    cv_results = pd.DataFrame(random_search.cv_results_)
    cv_results.to_csv(os.path.join(paramDir, 'random_search_full_results.csv'), index=False)

    # Save best parameters
    results_df = pd.DataFrame([best_params])
    results_df['best_score'] = best_score
    results_df.to_csv(os.path.join(paramDir, 'random_search_best_params.csv'), index=False)

    # Extract optimal parameters
    n_opt = best_params['n_estimators']
    max_depth = best_params['max_depth']
    min_samples_split = best_params['min_samples_split']
    min_samples_leaf = best_params['min_samples_leaf']
    max_features = best_params['max_features']
    bootstrap = best_params['bootstrap']

    return n_opt, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap
