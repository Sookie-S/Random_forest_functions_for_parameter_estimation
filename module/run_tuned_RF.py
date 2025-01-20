#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Soukaina Timouma
@email: soukaina.timouma@well.OX.AC.UK
"""

#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#%%
#############################################################################################################

#### Random forest functions

#############################################################################################################


def run_tuned_RF(
    X, y, testSize, n_opt, max_depth_opt, min_samples_split_opt, min_samples_leaf_opt,
    max_features_opt, Bootstrap, genes_alias, genes_nature, outdir, top_n_features=20
):
    """
    Run a tuned Random Forest classifier and analyse its results, including Precision-Recall curve, F1-Score, ROC curve, and AUC.

    Parameters:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target labels.
        testSize (float): Test set proportion.
        n_opt (int): Number of trees.
        max_depth_opt (int): Maximum depth of trees.
        min_samples_split_opt (int): Minimum samples for splitting.
        min_samples_leaf_opt (int): Minimum samples for leaf nodes.
        max_features_opt (str or int): Maximum features to consider at each split.
        Bootstrap (bool): Whether to use bootstrap samples.
        genes_alias (dict): Mapping of genes to aliases.
        genes_nature (dict): Mapping of genes to their nature.
        outdir (str): Output directory for saving results.
        top_n_features (int): Number of top features to display and save. Default is 20.

    Returns:
        tuple: Predicted labels, trained classifier, importance DataFrame.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testSize, random_state=42
    )

    # Initialize and fit Random Forest model
    rf_classifier = RandomForestClassifier(
        n_estimators=n_opt,
        max_depth=max_depth_opt,
        min_samples_split=min_samples_split_opt,
        min_samples_leaf=min_samples_leaf_opt,
        max_features=max_features_opt,
        bootstrap=Bootstrap,
        random_state=42
    )
    rf_classifier.fit(X_train, y_train)

    # Predict on test set
    y_pred = rf_classifier.predict(X_test)
    y_prob = rf_classifier.predict_proba(X_test)[:, 1]  # Get probabilities for positive class

    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save classification report
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
    report_path = os.path.join(outdir, f'accuracy_classification_report_RF_n{n_opt}.csv')
    report_df.to_csv(report_path, index=False)

    # Feature importances
    importances = rf_classifier.feature_importances_
    importance_df = pd.DataFrame({
        'Gene': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Map genes to aliases and natures
    importance_df['Gene_alias'] = importance_df['Gene'].map(genes_alias).fillna('Unknown')
    importance_df['Gene_nature'] = importance_df['Gene'].map(genes_nature).fillna('Unknown')

    # Save full feature importance
    importance_path = os.path.join(outdir, f'gene_importances_RF_n{n_opt}.csv')
    importance_df.to_csv(importance_path, index=False)

    # Plot top N feature importances
    plt.figure(figsize=(10, 6))
    top_features = importance_df.head(top_n_features)
    plt.barh(top_features['Gene_alias'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n_features} Genes by Importance')
    plt.gca().invert_yaxis()  # Show most important features at the top
    plot_path = os.path.join(outdir, f'gene_importances_top{top_n_features}_RF_n{n_opt}.png')
    plt.savefig(plot_path)
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    pr_curve_path = os.path.join(outdir, f'precision_recall_curve_RF_n{n_opt}.png')
    plt.savefig(pr_curve_path)
    plt.close()

    # F1-Score
    f1 = f1_score(y_test, y_pred)
    print(f'F1-Score: {f1:.4f}')
    
    # Save F1-Score to file
    f1_path = os.path.join(outdir, f'f1_score_RF_n{n_opt}.txt')
    with open(f1_path, 'w') as f:
        f.write(f'F1-Score: {f1:.4f}\n')

    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    roc_curve_path = os.path.join(outdir, f'roc_curve_RF_n{n_opt}.png')
    plt.savefig(roc_curve_path)
    plt.close()

    # Save AUC score
    auc_path = os.path.join(outdir, f'auc_score_RF_n{n_opt}.txt')
    with open(auc_path, 'w') as f:
        f.write(f'AUC Score: {auc:.4f}\n')

    return y_pred, rf_classifier, importance_df
