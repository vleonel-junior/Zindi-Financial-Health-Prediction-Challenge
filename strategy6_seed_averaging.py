"""
Strategy 6: Seed Averaging (Variance Reduction)
Attempts to bridge the gap to 0.91+ by averaging models trained on multiple seeds.
Reduces variance and improves generalization without adding model complexity.

Configuration:
- 5 Seeds for LGBM (V1 config)
- 5 Seeds for TabPFN (V1 config)
- Averaging probabilities before weighted ensemble
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

base_seed = 42
seeds = [42, 2024, 123, 777, 999]
TARGET_COL = 'Target'
ID_COL = 'ID'

def preprocess_for_lgbm(train_df, test_df):
    """Preprocessing for LGBM: Label Encoding + Basic Features"""
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
    
    train['income_per_age'] = train['personal_income'] / (train['owner_age'] + 1)
    test['income_per_age'] = test['personal_income'] / (test['owner_age'] + 1)
    
    return train, test, target

def preprocess_for_tabpfn(train_df, test_df):
    """Preprocessing for TabPFN V3: RAW data"""
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

def optimize_weights(lgbm_proba, tabpfn_proba, y_true):
    best_f1 = 0
    best_weight = 0.5
    
    print("\n🔍 Optimizing ensemble weights (Seed Averaged Probs)...")
    for weight_lgbm in np.linspace(0.3, 0.7, 100):
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
    
    print("🔧 Preprocessing...")
    X_lgbm, X_test_lgbm, y = preprocess_for_lgbm(train, test)
    X_tabpfn, X_test_tabpfn, _ = preprocess_for_tabpfn(train, test)
    
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    
    # Store aggregated probabilities
    lgbm_val_probs_sum = None
    tabpfn_val_probs_sum = None
    lgbm_test_probs_sum = np.zeros((len(test), len(target_le.classes_)))
    tabpfn_test_probs_sum = np.zeros((len(test), len(target_le.classes_)))
    
    # Validation split (Fixed seed for split stability to evaluate seeds)
    # Using base_seed for split ensures all models trained on same split
    X_lgbm_train, X_lgbm_val, X_tabpfn_train, X_tabpfn_val, y_train, y_val = train_test_split(
        X_lgbm, X_tabpfn, y_encoded, test_size=0.2, random_state=base_seed, stratify=y_encoded
    )
    
    if lgbm_val_probs_sum is None:
        lgbm_val_probs_sum = np.zeros((len(y_val), len(target_le.classes_)))
        tabpfn_val_probs_sum = np.zeros((len(y_val), len(target_le.classes_)))

    # ----------------------------------------------------------------
    # SEED AVERAGING LOOP
    # ----------------------------------------------------------------
    print(f"\n🌱 Starting Seed Averaging ({len(seeds)} seeds)...")
    
    for i, seed in enumerate(seeds):
        print(f"\n   --- Seed {i+1}/{len(seeds)}: {seed} ---")
        
        # 1. Train LGBM
        try:
            with open('best_params_lgbm.json') as f:
                lgbm_params = json.load(f)
        except:
            lgbm_params = {'n_estimators': 1000, 'learning_rate': 0.05}
            
        lgbm_params.update({'random_state': seed, 'verbose': -1, 'class_weight': 'balanced'})
        lgbm = LGBMClassifier(**lgbm_params)
        lgbm.fit(X_lgbm_train, y_train)
        
        lgbm_val_probs_sum += lgbm.predict_proba(X_lgbm_val)
        
        # Train full LGBM for test
        lgbm_full = LGBMClassifier(**lgbm_params)
        lgbm_full.fit(X_lgbm, y_encoded)
        lgbm_test_probs_sum += lgbm_full.predict_proba(X_test_lgbm)
        
        # 2. Train TabPFN
        # TabPFN seeds affect the sampling of the prior if applicable, 
        # but mostly internal logic. 'device' is not seeded, but we re-init.
        tabpfn = TabPFNClassifier(device=DEVICE) 
        # Note: TabPFN-2.5 might not have explicit random_state param exposed easily in __init__
        # but fitting it multiple times might yield slight variations if any stochasticity exists.
        # If deterministic, this just adds compute. Assuming slight variance.
        tabpfn.fit(X_tabpfn_train.values, y_train)
        tabpfn_val_probs_sum += tabpfn.predict_proba(X_tabpfn_val.values)
        
        tabpfn_full = TabPFNClassifier(device=DEVICE)
        tabpfn_full.fit(X_tabpfn.values, y_encoded)
        tabpfn_test_probs_sum += tabpfn_full.predict_proba(X_test_tabpfn.values)

    # Average probabilities
    print("\n➗ Averaging Probabilities...")
    lgbm_val_avg = lgbm_val_probs_sum / len(seeds)
    tabpfn_val_avg = tabpfn_val_probs_sum / len(seeds)
    
    lgbm_test_avg = lgbm_test_probs_sum / len(seeds)
    tabpfn_test_avg = tabpfn_test_probs_sum / len(seeds)
    
    # Optimize weights on averaged probabilities
    opt_weight_lgbm = optimize_weights(lgbm_val_avg, tabpfn_val_avg, y_val)
    opt_weight_tabpfn = 1 - opt_weight_lgbm
    
    # Ensemble Validation
    ensemble_val_proba = opt_weight_lgbm * lgbm_val_avg + opt_weight_tabpfn * tabpfn_val_avg
    ensemble_val_preds = ensemble_val_proba.argmax(axis=1)
    val_f1 = f1_score(y_val, ensemble_val_preds, average='macro')
    
    print(f"\n📋 Seed Averaged Ensemble Validation:")
    print(f"   F1 Score: {val_f1:.4f}")
    print(classification_report(y_val, ensemble_val_preds, target_names=target_le.classes_))
    
    # Ensemble Test
    ensemble_test_proba = opt_weight_lgbm * lgbm_test_avg + opt_weight_tabpfn * tabpfn_test_avg
    ensemble_preds = ensemble_test_proba.argmax(axis=1)
    final_labels = target_le.inverse_transform(ensemble_preds)
    
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: final_labels})
    submission.to_csv('submission_seed_avg.csv', index=False)
    
    print("\n✅ Saved to submission_seed_avg.csv")
    print(f"   Prediction distribution: {pd.Series(final_labels).value_counts().to_dict()}")

if __name__ == "__main__":
    main()
