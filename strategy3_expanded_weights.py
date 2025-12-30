"""
Strategy 3: Expanded Weight Optimization
Explores full weight space (0.0-1.0) with more iterations
Also optimizes number of models in ensemble (2, 3, or 4 models)
Expected gain: +0.003-0.008
"""
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
import warnings
import os
from itertools import product

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

def preprocess_standard(train_df, test_df):
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
    
    return train, test, target

def preprocess_raw(train_df, test_df):
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

def optimize_weights_expanded(probas_dict, y_true, n_trials=500):
    """
    Exhaustive weight optimization across all model combinations
    probas_dict: {'lgbm': proba_array, 'xgb': ..., 'cat': ..., 'tabpfn': ...}
    """
    best_f1 = 0
    best_weights = None
    best_models = None
    
    model_names = list(probas_dict.keys())
    print(f"\n🔍 Exhaustive weight optimization ({n_trials} trials per combination)...")
    print(f"   Testing all 2-4 model combinations from: {model_names}")
    
    # Test all combinations of 2, 3, and 4 models
    for n_models in [2, 3, 4]:
        from itertools import combinations
        for model_combo in combinations(model_names, n_models):
            # Random search for weights
            for _ in range(n_trials // 10):  # Divide trials among combinations
                # Generate random weights
                weights = np.random.dirichlet(np.ones(n_models))
                
                # Weighted average
                ensemble_proba = sum(weights[i] * probas_dict[model] 
                                    for i, model in enumerate(model_combo))
                preds = ensemble_proba.argmax(axis=1)
                
                f1 = f1_score(y_true, preds, average='macro')
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_weights = weights
                    best_models = model_combo
    
    print(f"   Best models: {best_models}")
    print(f"   Best weights: {dict(zip(best_models, best_weights))}")
    print(f"   Best CV F1: {best_f1:.4f}")
    
    return best_models, best_weights, best_f1

def main():
    print("\n📚 Loading Data...")
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv')
    test_ids = test[ID_COL]
    
    # Preprocess
    print("🔧 Preprocessing...")
    X_std, X_test_std, y = preprocess_standard(train, test)
    X_raw, X_test_raw, _ = preprocess_raw(train, test)
    
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    
    # Validation split
    X_std_train, X_std_val, X_raw_train, X_raw_val, y_train, y_val = train_test_split(
        X_std, X_raw, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
    )
    
    print(f"\n📊 Training 4 base models...")
    
    # Train base models
    models_val_proba = {}
    models_test_proba = {}
    
    # LGBM
    print("   Training LGBM...")
    try:
        with open('best_params_lgbm.json') as f:
            lgbm_params = json.load(f)
    except:
        lgbm_params = {'n_estimators': 1000, 'learning_rate': 0.05}
    lgbm_params.update({'random_state': SEED, 'verbose': -1, 'class_weight': 'balanced'})
    lgbm = LGBMClassifier(**lgbm_params)
    lgbm.fit(X_std_train, y_train)
    models_val_proba['lgbm'] = lgbm.predict_proba(X_std_val)
    
    lgbm_full = LGBMClassifier(**lgbm_params)
    lgbm_full.fit(X_std, y_encoded)
    models_test_proba['lgbm'] = lgbm_full.predict_proba(X_test_std)
    
    # XGBoost
    print("   Training XGBoost...")
    try:
        with open('best_params_xgboost.json') as f:
            xgb_params = json.load(f)
    except:
        xgb_params = {'n_estimators': 500, 'learning_rate': 0.1}
    xgb_params.update({'random_state': SEED, 'verbosity': 0})
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_std_train, y_train)
    models_val_proba['xgb'] = xgb.predict_proba(X_std_val)
    
    xgb_full = XGBClassifier(**xgb_params)
    xgb_full.fit(X_std, y_encoded)
    models_test_proba['xgb'] = xgb_full.predict_proba(X_test_std)
    
    # CatBoost
    print("   Training CatBoost...")
    try:
        with open('best_params_catboost.json') as f:
            cat_params = json.load(f)
    except:
        cat_params = {'iterations': 500, 'learning_rate': 0.1}
    cat_params.update({'random_state': SEED, 'verbose': 0})
    cat = CatBoostClassifier(**cat_params)
    cat.fit(X_std_train, y_train)
    models_val_proba['cat'] = cat.predict_proba(X_std_val)
    
    cat_full = CatBoostClassifier(**cat_params)
    cat_full.fit(X_std, y_encoded)
    models_test_proba['cat'] = cat_full.predict_proba(X_test_std)
    
    # TabPFN
    print("   Training TabPFN...")
    tabpfn = TabPFNClassifier(device=DEVICE)
    tabpfn.fit(X_raw_train.values, y_train)
    models_val_proba['tabpfn'] = tabpfn.predict_proba(X_raw_val.values)
    
    tabpfn_full = TabPFNClassifier(device=DEVICE)
    tabpfn_full.fit(X_raw.values, y_encoded)
    models_test_proba['tabpfn'] = tabpfn_full.predict_proba(X_test_raw.values)
    
    # Optimize weights
    best_models, best_weights, val_f1 = optimize_weights_expanded(models_val_proba, y_val, n_trials=500)
    
    # Ensemble test predictions
    ensemble_test_proba = sum(best_weights[i] * models_test_proba[model] 
                              for i, model in enumerate(best_models))
    ensemble_preds = ensemble_test_proba.argmax(axis=1)
    ensemble_labels = target_le.inverse_transform(ensemble_preds)
    
    # Submission
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: ensemble_labels})
    submission.to_csv('submission_expanded_weights.csv', index=False)
    
    print("\n✅ Saved to submission_expanded_weights.csv")
    print(f"   Prediction distribution: {pd.Series(ensemble_labels).value_counts().to_dict()}")
    print(f"\n🎯 Expected gain: +0.003-0.008 via exhaustive optimization")

if __name__ == "__main__":
    main()
