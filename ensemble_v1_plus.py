"""
Ensemble V1-Plus: Ultra-Optimized Simple Ensemble
Refined version of best performer (V1: 0.8978) with 3 targeted optimizations:
1. Ultra-fine weight search (0.40-0.55 instead of 0.3-0.7)
2. Add XGBoost as 3rd model (LGBM + TabPFN + XGBoost)
3. Threshold optimization for "High" class prediction
Target: 0.91+ F1 Score
"""
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
import warnings
import os

warnings.filterwarnings('ignore')

# GPU detection
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {DEVICE.upper()}")
except ImportError:
    DEVICE = 'cpu'

# HF Auth
try:
    import huggingface_hub
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        huggingface_hub.login(token=hf_token, add_to_git_credential=False)
except:
    pass

SEED = 42
TARGET_COL = 'Target'
ID_COL = 'ID'

def preprocess_for_trees(train_df, test_df):
    """Preprocessing for LGBM/XGBoost: Label Encoding"""
    train = train_df.drop(columns=[ID_COL, TARGET_COL])
    test = test_df.drop(columns=[ID_COL])
    target = train_df[TARGET_COL]
    
    for col in train.columns:
        if train[col].dtype == 'object':
            train[col] = train[col].fillna("Unknown")
            test[col] = test[col].fillna("Unknown")
        else:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)
    
    cat_cols = train.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
    
    # Add basic feature
    train['income_per_age'] = train['personal_income'] / (train['owner_age'] + 1)
    test['income_per_age'] = test['personal_income'] / (test['owner_age'] + 1)
    
    return train, test, target

def preprocess_for_tabpfn(train_df, test_df):
    """Preprocessing for TabPFN: RAW data"""
    train = train_df.drop(columns=[ID_COL, TARGET_COL])
    test = test_df.drop(columns=[ID_COL])
    target = train_df[TARGET_COL]
    
    for col in train.columns:
        if train[col].dtype == 'object':
            train[col] = train[col].fillna("Unknown")
            test[col] = test[col].fillna("Unknown")
        else:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)
    
    return train, test, target

def optimize_weights_ultra_fine(probas_dict, y_true, n_trials=1000):
    """
    OPTIMIZATION 1: Ultra-fine weight search
    Focus on narrow range around V1 optimal (0.46 LGBM, 0.54 TabPFN)
    """
    best_f1 = 0
    best_weights = None
    
    print("\n🔍 Ultra-Fine Weight Optimization (1000 trials)...")
    print("   Narrow range: LGBM [0.40-0.55], XGB [0.15-0.35], TabPFN [0.30-0.50]")
    
    for _ in range(n_trials):
        # Sample from narrow ranges
        w_lgbm = np.random.uniform(0.40, 0.55)
        w_xgb = np.random.uniform(0.15, 0.35)
        w_tabpfn = 1 - w_lgbm - w_xgb
        
        # Skip if tabpfn out of range
        if w_tabpfn < 0.30 or w_tabpfn > 0.50:
            continue
        
        # Weighted average
        ensemble_proba = (w_lgbm * probas_dict['lgbm'] + 
                         w_xgb * probas_dict['xgb'] + 
                         w_tabpfn * probas_dict['tabpfn'])
        preds = ensemble_proba.argmax(axis=1)
        
        f1 = f1_score(y_true, preds, average='macro')
        
        if f1 > best_f1:
            best_f1 = f1
            best_weights = {'lgbm': w_lgbm, 'xgb': w_xgb, 'tabpfn': w_tabpfn}
    
    print(f"   Best weights: LGBM={best_weights['lgbm']:.4f}, XGB={best_weights['xgb']:.4f}, TabPFN={best_weights['tabpfn']:.4f}")
    print(f"   Best CV F1 (Macro): {best_f1:.4f}")
    
    return best_weights, best_f1

def optimize_threshold_for_high_class(proba, y_true, target_le):
    """
    OPTIMIZATION 3: Threshold optimization for "High" class
    Find optimal decision threshold to maximize F1 for High class
    """
    print("\n🎯 Optimizing threshold for 'High' class...")
    
    high_idx = list(target_le.classes_).index('High')
    
    best_f1_high = 0
    best_threshold = 0.33  # Default 1/3 for 3 classes
    
    # Try different thresholds for High class
    for threshold in np.linspace(0.25, 0.50, 50):
        # Adjusted prediction: prioritize High if proba > threshold
        preds_adjusted = proba.argmax(axis=1).copy()
        high_mask = proba[:, high_idx] > threshold
        preds_adjusted[high_mask] = high_idx
        
        # Calculate F1 for High class only
        from sklearn.metrics import precision_recall_fscore_support
        _, _, f1_per_class, _ = precision_recall_fscore_support(y_true, preds_adjusted, average=None, zero_division=0)
        f1_high = f1_per_class[high_idx]
        
        if f1_high > best_f1_high:
            best_f1_high = f1_high
            best_threshold = threshold
    
    print(f"   Best threshold for High: {best_threshold:.3f}")
    print(f"   F1 for High class: {best_f1_high:.4f}")
    
    return best_threshold

def main():
    print("\n📚 Loading Data...")
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv')
    test_ids = test[ID_COL]
    
    # Preprocess
    print("🔧 Preprocessing...")
    X_trees, X_test_trees, y = preprocess_for_trees(train, test)
    X_tabpfn, X_test_tabpfn, _ = preprocess_for_tabpfn(train, test)
    
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    
    print(f"\n📊 Dataset: {len(y)} samples, {len(target_le.classes_)} classes")
    
    # Validation split
    X_trees_train, X_trees_val, X_tabpfn_train, X_tabpfn_val, y_train, y_val = train_test_split(
        X_trees, X_tabpfn, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
    )
    
    print("\n🚀 Training 3 base models...")
    
    # LGBM
    print("   Training LGBM...")
    try:
        with open('best_params_lgbm.json') as f:
            lgbm_params = json.load(f)
    except:
        lgbm_params = {'n_estimators': 1000, 'learning_rate': 0.05}
    lgbm_params.update({'random_state': SEED, 'verbose': -1, 'class_weight': 'balanced'})
    lgbm = LGBMClassifier(**lgbm_params)
    lgbm.fit(X_trees_train, y_train)
    lgbm_val_proba = lgbm.predict_proba(X_trees_val)
    
    # XGBoost (OPTIMIZATION 2: Add as 3rd model)
    print("   Training XGBoost...")
    try:
        with open('best_params_xgboost.json') as f:
            xgb_params = json.load(f)
    except:
        xgb_params = {'n_estimators': 500, 'learning_rate': 0.1}
    xgb_params.update({'random_state': SEED, 'verbosity': 0})
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_trees_train, y_train)
    xgb_val_proba = xgb.predict_proba(X_trees_val)
    
    # TabPFN
    print(f"   Training TabPFN on {DEVICE.upper()}...")
    tabpfn = TabPFNClassifier(device=DEVICE)
    tabpfn.fit(X_tabpfn_train.values, y_train)
    tabpfn_val_proba = tabpfn.predict_proba(X_tabpfn_val.values)
    
    # OPTIMIZATION 1: Ultra-fine weight search
    probas_val = {
        'lgbm': lgbm_val_proba,
        'xgb': xgb_val_proba,
        'tabpfn': tabpfn_val_proba
    }
    optimal_weights, val_f1 = optimize_weights_ultra_fine(probas_val, y_val, n_trials=1000)
    
    # Get ensemble validation probabilities
    ensemble_val_proba = (optimal_weights['lgbm'] * lgbm_val_proba + 
                          optimal_weights['xgb'] * xgb_val_proba + 
                          optimal_weights['tabpfn'] * tabpfn_val_proba)
    
    # OPTIMIZATION 3: Threshold optimization for High class
    optimal_threshold_high = optimize_threshold_for_high_class(ensemble_val_proba, y_val, target_le)
    
    # Apply threshold optimization
    high_idx = list(target_le.classes_).index('High')
    ensemble_val_preds = ensemble_val_proba.argmax(axis=1)
    high_mask = ensemble_val_proba[:, high_idx] > optimal_threshold_high
    ensemble_val_preds[high_mask] = high_idx
    
    # Validation report
    final_val_f1 = f1_score(y_val, ensemble_val_preds, average='macro')
    
    print(f"\n📋 V1-Plus Validation Report:")
    print(f"   Validation F1 (Macro): {final_val_f1:.4f}")
    print(classification_report(y_val, ensemble_val_preds, target_names=target_le.classes_))
    
    # Train on full data for submission
    print("\n🔄 Retraining on full dataset...")
    lgbm_full = LGBMClassifier(**lgbm_params)
    lgbm_full.fit(X_trees, y_encoded)
    
    xgb_full = XGBClassifier(**xgb_params)
    xgb_full.fit(X_trees, y_encoded)
    
    tabpfn_full = TabPFNClassifier(device=DEVICE)
    tabpfn_full.fit(X_tabpfn.values, y_encoded)
    
    # Test predictions
    print("\n🔮 Generating test predictions...")
    lgbm_test_proba = lgbm_full.predict_proba(X_test_trees)
    xgb_test_proba = xgb_full.predict_proba(X_test_trees)
    tabpfn_test_proba = tabpfn_full.predict_proba(X_test_tabpfn.values)
    
    # Ensemble with optimal weights
    ensemble_test_proba = (optimal_weights['lgbm'] * lgbm_test_proba + 
                           optimal_weights['xgb'] * xgb_test_proba + 
                           optimal_weights['tabpfn'] * tabpfn_test_proba)
    
    # Apply threshold optimization
    ensemble_test_preds = ensemble_test_proba.argmax(axis=1)
    high_mask_test = ensemble_test_proba[:, high_idx] > optimal_threshold_high
    ensemble_test_preds[high_mask_test] = high_idx
    
    ensemble_labels = target_le.inverse_transform(ensemble_test_preds)
    
    # Submission
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: ensemble_labels})
    submission.to_csv('submission_v1_plus_optimized.csv', index=False)
    
    print("\n✅ Saved to submission_v1_plus_optimized.csv")
    print(f"   Prediction distribution: {pd.Series(ensemble_labels).value_counts().to_dict()}")
    print(f"\n🎯 V1-Plus Optimizations Applied:")
    print(f"   ✅ Ultra-fine weight search (narrow ranges)")
    print(f"   ✅ Added XGBoost as 3rd model")
    print(f"   ✅ Threshold optimization for 'High' class (threshold={optimal_threshold_high:.3f})")
    print(f"\n   Expected: 0.900-0.915 (target: 0.91+)")

if __name__ == "__main__":
    main()
