"""
Ensemble LGBM + TabPFN V2: Both with RAW Categorical Handling
LGBM uses native categorical_feature parameter (no Label Encoding)
TabPFN uses internal preprocessing (V3 proven best)
"""
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from lightgbm import LGBMClassifier
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

def ultra_raw_preprocessing_shared(train_df, test_df):
    """
    ULTRA-RAW: Same preprocessing for both LGBM and TabPFN
    Only fill missing, no encoding at all
    """
    train = train_df.drop(columns=[ID_COL, TARGET_COL])
    test = test_df.drop(columns=[ID_COL])
    target = train_df[TARGET_COL]
    
    # Only fill missing
    for col in train.columns:
        if train[col].dtype == 'object':
            train[col] = train[col].fillna("Unknown")
            test[col] = test[col].fillna("Unknown")
        else:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)
    
    # Identify categorical columns for LGBM
    cat_cols = train.select_dtypes(include=['object']).columns.tolist()
    
    return train, test, target, cat_cols

def optimize_weights(lgbm_proba, tabpfn_proba, y_true, n_trials=100):
    """
    Find optimal weights via grid search
    """
    best_f1 = 0
    best_weight = 0.5
    
    print("\n🔍 Optimizing ensemble weights...")
    for weight_lgbm in np.linspace(0.3, 0.7, n_trials):
        weight_tabpfn = 1 - weight_lgbm
        
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
    
    # Shared ultra-raw preprocessing
    print("🔧 Ultra-Raw Preprocessing (shared for both models)...")
    print("   LGBM: Native categorical handling via categorical_feature")
    print("   TabPFN: Internal preprocessing (V3)")
    
    X, X_test, y, cat_cols = ultra_raw_preprocessing_shared(train, test)
    
    print(f"\n   Categorical columns ({len(cat_cols)}): {cat_cols[:5]}...")
    print(f"   Features: {X.shape[1]}")
    
    # Encode target
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    
    print(f"   Classes: {list(target_le.classes_)}")
    
    # Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
    )
    
    # Load or use default LGBM params
    print("\n🚀 Training LGBM with native categorical handling...")
    try:
        with open('best_params_lgbm.json', 'r') as f:
            lgbm_params = json.load(f)
        print(f"   Loaded params from best_params_lgbm.json")
    except:
        print("   Using default params")
        lgbm_params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1
        }
    
    lgbm_params.update({
        'random_state': SEED,
        'verbose': -1,
        'class_weight': 'balanced',
        'categorical_feature': cat_cols  # CRITICAL: Native categorical handling
    })
    
    lgbm = LGBMClassifier(**lgbm_params)
    lgbm.fit(X_train, y_train, categorical_feature=cat_cols)
    print("   ✅ LGBM trained (categorical_feature enabled)")
    
    # Train TabPFN V3 (raw)
    print(f"\n🚀 Training TabPFN V3 (RAW) on {DEVICE.upper()}...")
    tabpfn = TabPFNClassifier(device=DEVICE)
    tabpfn.fit(X_train.values, y_train)
    print("   ✅ TabPFN trained")
    
    # Validation predictions
    print("\n📊 Generating validation predictions...")
    lgbm_val_proba = lgbm.predict_proba(X_val)
    tabpfn_val_proba = tabpfn.predict_proba(X_val.values)
    
    # Optimize weights
    optimal_weight_lgbm = optimize_weights(lgbm_val_proba, tabpfn_val_proba, y_val)
    optimal_weight_tabpfn = 1 - optimal_weight_lgbm
    
    # Validate ensemble
    ensemble_val_proba = optimal_weight_lgbm * lgbm_val_proba + optimal_weight_tabpfn * tabpfn_val_proba
    ensemble_val_preds = ensemble_val_proba.argmax(axis=1)
    val_f1 = f1_score(y_val, ensemble_val_preds, average='macro')
    
    print(f"\n📋 Ensemble V2 (Both Raw) Validation Report:")
    print(f"   Validation F1 (Macro): {val_f1:.4f}")
    print(classification_report(y_val, ensemble_val_preds, target_names=target_le.classes_))
    
    # Train on full data
    print("\n🔄 Retraining on full dataset...")
    lgbm_full = LGBMClassifier(**lgbm_params)
    lgbm_full.fit(X, y_encoded, categorical_feature=cat_cols)
    
    tabpfn_full = TabPFNClassifier(device=DEVICE)
    tabpfn_full.fit(X.values, y_encoded)
    
    # Test predictions
    print("\n🔮 Generating test predictions...")
    lgbm_test_proba = lgbm_full.predict_proba(X_test)
    tabpfn_test_proba = tabpfn_full.predict_proba(X_test.values)
    
    # Ensemble
    ensemble_test_proba = optimal_weight_lgbm * lgbm_test_proba + optimal_weight_tabpfn * tabpfn_test_proba
    ensemble_preds = ensemble_test_proba.argmax(axis=1)
    ensemble_labels = target_le.inverse_transform(ensemble_preds)
    
    # Submission
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: ensemble_labels})
    submission.to_csv('submission_ensemble_v2_both_raw.csv', index=False)
    
    print("\n✅ Saved to submission_ensemble_v2_both_raw.csv")
    print(f"   Prediction distribution: {pd.Series(ensemble_labels).value_counts().to_dict()}")
    print(f"\n🎯 V2 Strategy: Both models use RAW categorical handling")
    print(f"   Expected gain: +0.005-0.010 vs V1 (less encoding noise)")

if __name__ == "__main__":
    main()
