"""
Ensemble LGBM + TabPFN V3 (Legal Manual Ensemble - NOT AutoML)
Combines best models with weighted probability averaging
Target: 0.91+ F1 Score on Zindi leaderboard
"""
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
import warnings
import os

warnings.filterwarnings('ignore')

# GPU detection for TabPFN
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {DEVICE.upper()}")
except ImportError:
    DEVICE = 'cpu'

# HuggingFace Auth for TabPFN
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

def preprocess_for_lgbm(train_df, test_df):
    """
    Preprocessing for LGBM: Label Encoding + Basic Features
    """
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
    
    # Label encode categoricals
    cat_cols = train.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
    
    # Add engineered features
    train['income_per_age'] = train['personal_income'] / (train['owner_age'] + 1)
    test['income_per_age'] = test['personal_income'] / (test['owner_age'] + 1)
    
    return train, test, target

def preprocess_for_tabpfn(train_df, test_df):
    """
    Preprocessing for TabPFN V3: RAW data (minimal preprocessing)
    """
    train = train_df.drop(columns=[ID_COL, TARGET_COL])
    test = test_df.drop(columns=[ID_COL])
    target = train_df[TARGET_COL]
    
    # Only fill missing - NO encoding
    for col in train.columns:
        if train[col].dtype == 'object':
            train[col] = train[col].fillna("Unknown")
            test[col] = test[col].fillna("Unknown")
        else:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)
    
    return train, test, target

def optimize_weights(lgbm_proba, tabpfn_proba, y_true, n_trials=100):
    """
    Find optimal weights for ensemble using grid search
    """
    best_f1 = 0
    best_weight = 0.5
    
    print("\n🔍 Optimizing ensemble weights...")
    for weight_lgbm in np.linspace(0.3, 0.7, n_trials):
        weight_tabpfn = 1 - weight_lgbm
        
        # Weighted average
        ensemble_proba = weight_lgbm * lgbm_proba + weight_tabpfn * tabpfn_proba
        preds = ensemble_proba.argmax(axis=1)
        
        f1 = f1_score(y_true, preds, average='macro')
        
        if f1 > best_f1:
            best_f1 = f1
            best_weight = weight_lgbm
    
    print(f"   Best weights: LGBM={best_weight:.3f}, TabPFN={1-best_weight:.3f}")
    print(f"   Best CV F1: {best_f1:.4f}")
    
    return best_weight

def main():
    print("\n📚 Loading Data...")
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv')
    test_ids = test[ID_COL]
    
    # Preprocess for both models
    print("🔧 Preprocessing for LGBM...")
    X_lgbm, X_test_lgbm, y = preprocess_for_lgbm(train, test)
    
    print("🔧 Preprocessing for TabPFN (RAW)...")
    X_tabpfn, X_test_tabpfn, _ = preprocess_for_tabpfn(train, test)
    
    # Encode target
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    
    print(f"\n📊 Dataset: {len(y)} samples, {len(target_le.classes_)} classes")
    
    # Validation split
    X_lgbm_train, X_lgbm_val, X_tabpfn_train, X_tabpfn_val, y_train, y_val = train_test_split(
        X_lgbm, X_tabpfn, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
    )
    
    # Load best LGBM params
    print("\n🚀 Training LGBM with best params...")
    try:
        with open('best_params_lgbm.json', 'r') as f:
            lgbm_params = json.load(f)
        print(f"   Loaded params from best_params_lgbm.json")
    except:
        print("   Using default params (best_params_lgbm.json not found)")
        lgbm_params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'class_weight': 'balanced'
        }
    
    lgbm_params['random_state'] = SEED
    lgbm_params['verbose'] = -1
    
    lgbm = LGBMClassifier(**lgbm_params)
    lgbm.fit(X_lgbm_train, y_train)
    print("   ✅ LGBM trained")
    
    # Train TabPFN V3 (raw data)
    print(f"\n🚀 Training TabPFN V3 (RAW) on {DEVICE.upper()}...")
    tabpfn = TabPFNClassifier(device=DEVICE)
    tabpfn.fit(X_tabpfn_train.values, y_train)
    print("   ✅ TabPFN trained")
    
    # Get validation probabilities
    print("\n📊 Generating validation predictions...")
    lgbm_val_proba = lgbm.predict_proba(X_lgbm_val)
    tabpfn_val_proba = tabpfn.predict_proba(X_tabpfn_val.values)
    
    # Optimize weights
    optimal_weight_lgbm = optimize_weights(lgbm_val_proba, tabpfn_val_proba, y_val)
    optimal_weight_tabpfn = 1 - optimal_weight_lgbm
    
    # Validate ensemble
    ensemble_val_proba = optimal_weight_lgbm * lgbm_val_proba + optimal_weight_tabpfn * tabpfn_val_proba
    ensemble_val_preds = ensemble_val_proba.argmax(axis=1)
    val_f1 = f1_score(y_val, ensemble_val_preds, average='macro')
    
    print(f"\n📋 Ensemble Validation Report:")
    print(f"   Validation F1 (Macro): {val_f1:.4f}")
    print(classification_report(y_val, ensemble_val_preds, target_names=target_le.classes_))
    
    # Train on full data for final submission
    print("\n🔄 Retraining on full dataset...")
    lgbm_full = LGBMClassifier(**lgbm_params)
    lgbm_full.fit(X_lgbm, y_encoded)
    
    tabpfn_full = TabPFNClassifier(device=DEVICE)
    tabpfn_full.fit(X_tabpfn.values, y_encoded)
    
    # Generate test predictions
    print("\n🔮 Generating test predictions...")
    lgbm_test_proba = lgbm_full.predict_proba(X_test_lgbm)
    tabpfn_test_proba = tabpfn_full.predict_proba(X_test_tabpfn.values)
    
    # Ensemble test predictions
    ensemble_test_proba = optimal_weight_lgbm * lgbm_test_proba + optimal_weight_tabpfn * tabpfn_test_proba
    ensemble_preds = ensemble_test_proba.argmax(axis=1)
    ensemble_labels = target_le.inverse_transform(ensemble_preds)
    
    # Create submission
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: ensemble_labels})
    submission.to_csv('submission_ensemble_lgbm_tabpfn.csv', index=False)
    
    print("\n✅ Saved to submission_ensemble_lgbm_tabpfn.csv")
    print(f"   Prediction distribution: {pd.Series(ensemble_labels).value_counts().to_dict()}")
    print(f"\n🎯 Expected LB Score: ~0.900-0.910 (based on component scores)")
    print(f"   LGBM: 0.893, TabPFN V3: 0.891, Ensemble CV: {val_f1:.4f}")

if __name__ == "__main__":
    main()
