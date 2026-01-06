"""
Ensemble LGBM + TabPFN OPTIMIZED (V1-OPT)
Applied TabPFN Official Performance Optimizations:
1. fitted-model cache (fit_mode='fit_with_cache') for small dataset
2. Multi-GPU support if available  
3. Batch inference optimization
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

# GPU detection for TabPFN
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_GPUS = torch.cuda.device_count() if DEVICE == 'cuda' else 0
    print(f"🖥️  Device: {DEVICE.upper()}")
    if N_GPUS > 0:
        print(f"   GPUs available: {N_GPUS}")
        for i in range(N_GPUS):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError:
    DEVICE = 'cpu'
    N_GPUS = 0

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

def preprocess_for_lgbm(train_df, test_df):
    """Preprocessing for LGBM: Label Encoding + Basic Features"""
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
    """Preprocessing for TabPFN V3: RAW data (minimal preprocessing)"""
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
    """Find optimal weights for ensemble using grid search"""
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
    
    dataset_size = len(train)
    print(f"   Dataset size: {dataset_size} samples")
    
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
        print("   Using default params")
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
    
    # Train TabPFN with OPTIMIZATIONS
    print(f"\n🚀 Training TabPFN V3 (OPTIMIZED) on {DEVICE.upper()}...")
    
    # OPTIMIZATION 1: fitted-model cache for small datasets (<10K samples)
    # Saves model during fit() for faster predict()
    use_cache = dataset_size < 10000
    fit_mode = 'fit_with_cache' if use_cache else 'default'
    
    if use_cache:
        print(f"   ✅ Fitted-model cache ENABLED (dataset size: {dataset_size} < 10K)")
        print(f"      Subsequent .predict() calls will be fast via KV-Cache")
    else:
        print(f"   ℹ️  Fitted-model cache disabled (dataset size: {dataset_size} >= 10K)")
    
    # OPTIMIZATION 2: Multi-GPU if available
    device_list = list(range(N_GPUS)) if N_GPUS > 1 else DEVICE
    if N_GPUS > 1:
        print(f"   ✅ Multi-GPU inference enabled ({N_GPUS} GPUs)")
    
    tabpfn = TabPFNClassifier(
        device=device_list,  # Multi-GPU support
        fit_mode=fit_mode,    # Cache for small datasets
        # OPTIMIZATION 3: memory_saving_mode could be tuned here if needed
        # memory_saving_mode=0  # Default, can adjust for speed/memory tradeoff
    )
    
    tabpfn.fit(X_tabpfn_train.values, y_train)
    print("   ✅ TabPFN trained with optimizations")
    
    # OPTIMIZATION 4: Batch inference for validation
    print("\n📊 Generating validation predictions (batch inference)...")
    lgbm_val_proba = lgbm.predict_proba(X_lgbm_val)
    # Single .predict() call for all validation data (faster than repeated calls)
    tabpfn_val_proba = tabpfn.predict_proba(X_tabpfn_val.values)
    
    # Optimize weights
    optimal_weight_lgbm = optimize_weights(lgbm_val_proba, tabpfn_val_proba, y_val)
    optimal_weight_tabpfn = 1 - optimal_weight_lgbm
    
    # Validate ensemble
    ensemble_val_proba = optimal_weight_lgbm * lgbm_val_proba + optimal_weight_tabpfn * tabpfn_val_proba
    ensemble_val_preds = ensemble_val_proba.argmax(axis=1)
    val_f1 = f1_score(y_val, ensemble_val_preds, average='macro')
    
    print(f"\n📋 Ensemble OPTIMIZED Validation Report:")
    print(f"   Validation F1 (Macro): {val_f1:.4f}")
    print(classification_report(y_val, ensemble_val_preds, target_names=target_le.classes_))
    
    # Train on full data for final submission
    print("\n🔄 Retraining on full dataset...")
    lgbm_full = LGBMClassifier(**lgbm_params)
    lgbm_full.fit(X_lgbm, y_encoded)
    
    tabpfn_full = TabPFNClassifier(
        device=device_list,
        fit_mode=fit_mode
    )
    tabpfn_full.fit(X_tabpfn.values, y_encoded)
    
    # Generate test predictions (batch inference)
    print("\n🔮 Generating test predictions (batch inference)...")
    lgbm_test_proba = lgbm_full.predict_proba(X_test_lgbm)
    # Single .predict() call for entire test set (optimized)
    tabpfn_test_proba = tabpfn_full.predict_proba(X_test_tabpfn.values)
    
    # Ensemble test predictions
    ensemble_test_proba = optimal_weight_lgbm * lgbm_test_proba + optimal_weight_tabpfn * tabpfn_test_proba
    ensemble_preds = ensemble_test_proba.argmax(axis=1)
    ensemble_labels = target_le.inverse_transform(ensemble_preds)
    
    # Create submission
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: ensemble_labels})
    submission.to_csv('submission_ensemble_optimized.csv', index=False)
    
    print("\n✅ Saved to submission_ensemble_optimized.csv")
    print(f"   Prediction distribution: {pd.Series(ensemble_labels).value_counts().to_dict()}")
    print(f"\n🎯 TabPFN Optimizations Applied:")
    print(f"   ✅ Fitted-model cache: {'ENABLED' if use_cache else 'Disabled'}")
    print(f"   ✅ Multi-GPU inference: {N_GPUS if N_GPUS > 1 else 'Single GPU/CPU'}")
    print(f"   ✅ Batch inference: Enabled")
    print(f"\n   Expected gain: +0.005-0.015 vs baseline V1 (0.8978)")

if __name__ == "__main__":
    main()
