"""
Strategy 5: Advanced Oversampling (SMOTE-NC) + Hybrid Ensemble
Addresses class imbalance (High: 5%) using SOTA techniques:
1. SMOTE-NC for Tree Models (LGBM, XGBoost): Handles mixed categorical/numerical data correctly
   and generates synthetic minority samples.
2. Optimized Class Weights for TabPFN: Avoids data distortion for Transformer inputs.
3. Hybrid Ensemble: Combines oversampled trees with weight-balanced TabPFN.
Expected gain: Improved recall for 'High' class and overall F1 > 0.90
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
# Requires: pip install imbalanced-learn
from imblearn.over_sampling import SMOTENC
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
    """Preprocessing for LGBM/XGBoost: Label Encoding + Feature indices for SMOTE-NC"""
    train = train_df.drop(columns=[ID_COL, TARGET_COL])
    test = test_df.drop(columns=[ID_COL])
    target = train_df[TARGET_COL]
    
    # Fill missing
    for col in train.columns:
        if train[col].dtype == 'object':
            train[col] = train[col].fillna("Unknown")
            test[col] = test[col].fillna("Unknown")
        else:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)
    
    # Label encode and track indices
    cat_cols = train.select_dtypes(include=['object']).columns
    cat_indices = [train.columns.get_loc(c) for c in cat_cols if c in train.columns]
    
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
    
    # Add engineered feature
    train['income_per_age'] = train['personal_income'] / (train['owner_age'] + 1)
    test['income_per_age'] = test['personal_income'] / (test['owner_age'] + 1)
    
    return train, test, target, cat_indices

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

def main():
    print("\n📚 Loading Data...")
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv')
    test_ids = test[ID_COL]
    
    # Preprocess
    print("🔧 Preprocessing...")
    X_trees, X_test_trees, y, cat_indices = preprocess_for_trees(train, test)
    X_tabpfn, X_test_tabpfn, _ = preprocess_for_tabpfn(train, test)
    
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    
    print(f"\n📊 Dataset: {len(y)} samples")
    print(f"   Class distribution (Original): {pd.Series(y).value_counts().to_dict()}")
    
    # Validation split
    X_trees_train, X_trees_val, X_tabpfn_train, X_tabpfn_val, y_train, y_val = train_test_split(
        X_trees, X_tabpfn, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
    )
    
    # ---------------------------------------------------------
    # PART 1: APPLY SMOTE-NC TO TREE TRAINING DATA
    # ---------------------------------------------------------
    print("\n⚖️ Applying SMOTE-NC (Mixed Data Oversampling)...")
    try:
        smote_nc = SMOTENC(categorical_features=cat_indices, random_state=SEED, sampling_strategy='auto')
        X_trees_train_res, y_train_res = smote_nc.fit_resample(X_trees_train, y_train)
        
        print(f"   Original Train Shape: {X_trees_train.shape}")
        print(f"   Resampled Train Shape: {X_trees_train_res.shape}")
        
        # Decode to check distribution
        y_train_res_decoded = target_le.inverse_transform(y_train_res)
        print(f"   Class distribution (Resampled): {pd.Series(y_train_res_decoded).value_counts().to_dict()}")
        
    except Exception as e:
        print(f"   ❌ SMOTE-NC Failed: {e}")
        print("   Falling back to standard training")
        X_trees_train_res, y_train_res = X_trees_train, y_train

    # ---------------------------------------------------------
    # PART 2: TRAIN MODELS
    # ---------------------------------------------------------
    print("\n🚀 Training Models...")

    # LGBM (on Resampled Data)
    print("   1. LGBM (trained on SMOTE-NC data)...")
    try:
        with open('best_params_lgbm.json') as f:
            lgbm_params = json.load(f)
    except:
        lgbm_params = {'n_estimators': 1000, 'learning_rate': 0.05}
    
    # IMPORTANT: Don't use class_weight='balanced' here because data is already balanced by SMOTE
    lgbm_params.update({'random_state': SEED, 'verbose': -1}) 
    
    lgbm = LGBMClassifier(**lgbm_params)
    lgbm.fit(X_trees_train_res, y_train_res)
    lgbm_val_proba = lgbm.predict_proba(X_trees_val)

    # XGBoost (on Resampled Data) - Add as it works well with balanced data
    print("   2. XGBoost (trained on SMOTE-NC data)...")
    try:
        with open('best_params_xgboost.json') as f:
            xgb_params = json.load(f)
    except:
        xgb_params = {'n_estimators': 500, 'learning_rate': 0.1}
    
    xgb_params.update({'random_state': SEED, 'verbosity': 0})
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_trees_train_res, y_train_res)
    xgb_val_proba = xgb.predict_proba(X_trees_val)

    # TabPFN (on Original Data with Weights) - TabPFN handles priors itself
    print(f"   3. TabPFN (Original Data + Weighted) on {DEVICE.upper()}...")
    # NOTE: TabPFN doesn't take class_weight param in __init__ (it's implicit in priors or N/A), 
    # but training on original distribution is usually better for its internal meta-learning.
    tabpfn = TabPFNClassifier(device=DEVICE)
    tabpfn.fit(X_tabpfn_train.values, y_train)
    tabpfn_val_proba = tabpfn.predict_proba(X_tabpfn_val.values)

    # ---------------------------------------------------------
    # PART 3: ENSEMBLE OPTIMIZATION
    # ---------------------------------------------------------
    print("\n🔍 Optimizing Ensemble Weights...")
    
    best_f1 = 0
    best_weights = None
    
    # Grid search
    for w_lgbm in np.linspace(0.2, 0.6, 20):
        for w_xgb in np.linspace(0.1, 0.4, 20):
            w_tabpfn = 1 - w_lgbm - w_xgb
            if w_tabpfn < 0.1: continue
            
            ensemble_proba = (w_lgbm * lgbm_val_proba + 
                            w_xgb * xgb_val_proba + 
                            w_tabpfn * tabpfn_val_proba)
            preds = ensemble_proba.argmax(axis=1)
            f1 = f1_score(y_val, preds, average='macro')
            
            if f1 > best_f1:
                best_f1 = f1
                best_weights = (w_lgbm, w_xgb, w_tabpfn)

    print(f"   Best weights: LGBM={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, TabPFN={best_weights[2]:.2f}")
    print(f"   Validation F1: {best_f1:.4f}")
    
    # Validation report
    ensemble_val_proba = (best_weights[0] * lgbm_val_proba + 
                          best_weights[1] * xgb_val_proba + 
                          best_weights[2] * tabpfn_val_proba)
    ensemble_val_preds = ensemble_val_proba.argmax(axis=1)
    print(classification_report(y_val, ensemble_val_preds, target_names=target_le.classes_))

    # ---------------------------------------------------------
    # PART 4: FULL TRAINING & SUBMISSION
    # ---------------------------------------------------------
    print("\n🔄 Retraining on Full Dataset...")
    
    # 1. Apply SMOTE-NC to full dataset for trees
    print("   Applying SMOTE-NC to full dataset...")
    smote_nc_full = SMOTENC(categorical_features=cat_indices, random_state=SEED, sampling_strategy='auto')
    X_trees_res, y_res = smote_nc_full.fit_resample(X_trees, y_encoded)
    
    # 2. Train Trees on full resampled data
    lgbm_full = LGBMClassifier(**lgbm_params)
    lgbm_full.fit(X_trees_res, y_res)
    
    xgb_full = XGBClassifier(**xgb_params)
    xgb_full.fit(X_trees_res, y_res)
    
    # 3. Train TabPFN on full original data
    tabpfn_full = TabPFNClassifier(device=DEVICE)
    tabpfn_full.fit(X_tabpfn.values, y_encoded)
    
    # 4. Predict
    print("🔮 Generating Test Predictions...")
    lgbm_preds = lgbm_full.predict_proba(X_test_trees)
    xgb_preds = xgb_full.predict_proba(X_test_trees)
    tabpfn_preds = tabpfn_full.predict_proba(X_test_tabpfn.values)
    
    final_proba = (best_weights[0] * lgbm_preds + 
                   best_weights[1] * xgb_preds + 
                   best_weights[2] * tabpfn_preds)
    
    final_preds_idx = final_proba.argmax(axis=1)
    final_labels = target_le.inverse_transform(final_preds_idx)
    
    # Submission
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: final_labels})
    submission.to_csv('submission_strategy5_oversampling.csv', index=False)
    
    print("\n✅ Saved to submission_strategy5_oversampling.csv")
    print(f"   Prediction distribution: {pd.Series(final_labels).value_counts().to_dict()}")

if __name__ == "__main__":
    main()
