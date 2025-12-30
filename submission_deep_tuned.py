"""
Strategy 7: Submission with Ultra-Deep Tuned Models
Loads best parameters from 'optuna_deep_tuning.py' output.
Training Pipeline:
1. Load best params for LGBM, XGB, CatBoost.
2. Train models on full dataset using these optimized params.
3. Train TabPFN (Raw).
4. Ensemble using optimized weights (tuned via OOF validation).
"""
import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
import warnings

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
N_FOLDS = 5

def preprocess_for_trees(train_df, test_df):
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

def load_best_params(model_name):
    filename = f'best_params_{model_name}_deep.json'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            print(f"   ✅ Loaded Deep Tuned Params for {model_name.upper()}")
            return json.load(f)
    print(f"   ⚠️ Tuning file {filename} not found. Using defaults.")
    return None

def main():
    print("\n📚 Loading Data...")
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv')
    test_ids = test[ID_COL]
    
    print("🔧 Preprocessing...")
    X_trees, X_test_trees, y = preprocess_for_trees(train, test)
    X_tabpfn, X_test_tabpfn, _ = preprocess_for_tabpfn(train, test)
    
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    
    # Init proba arrays
    n_classes = len(np.unique(y_encoded))
    train_probas = {
        'lgbm': np.zeros((len(train), n_classes)),
        'xgb': np.zeros((len(train), n_classes)),
        'cat': np.zeros((len(train), n_classes)),
        'tabpfn': np.zeros((len(train), n_classes))
    }
    test_probas = {
        'lgbm': np.zeros((len(test), n_classes)),
        'xgb': np.zeros((len(test), n_classes)),
        'cat': np.zeros((len(test), n_classes)),
        'tabpfn': np.zeros((len(test), n_classes))
    }
    
    # -------------------------------------------------------------
    # 1. Train Models with Deep Tuned Params (CV for Weights)
    # -------------------------------------------------------------
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    print("\n🔄 Training Models (CV for Weight Optimization)...")
    
    # Load Params
    params_lgbm = load_best_params('lgbm')
    params_xgb = load_best_params('xgboost')
    params_cat = load_best_params('catboost')
    
    # Apply defaults if missing
    if not params_lgbm: params_lgbm = {'n_estimators': 1000, 'learning_rate': 0.05, 'verbose': -1, 'random_state': SEED, 'class_weight': 'balanced'}
    if not params_xgb: params_xgb = {'n_estimators': 500, 'learning_rate': 0.1, 'verbosity': 0, 'random_state': SEED}
    if not params_cat: params_cat = {'iterations': 500, 'learning_rate': 0.1, 'verbose': 0, 'random_state': SEED}
    
    # CV Loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_trees, y_encoded)):
        print(f"   Fold {fold+1}/{N_FOLDS}")
        
        # Split
        X_tr_trees, X_val_trees = X_trees.iloc[train_idx], X_trees.iloc[val_idx]
        X_tr_tab, X_val_tab = X_tabpfn.iloc[train_idx], X_tabpfn.iloc[val_idx]
        y_tr = y_encoded[train_idx]
        
        # LGBM
        model_lgbm = LGBMClassifier(**params_lgbm)
        model_lgbm.fit(X_tr_trees, y_tr)
        train_probas['lgbm'][val_idx] = model_lgbm.predict_proba(X_val_trees)
        test_probas['lgbm'] += model_lgbm.predict_proba(X_test_trees) / N_FOLDS
        
        # XGB
        model_xgb = XGBClassifier(**params_xgb)
        model_xgb.fit(X_tr_trees, y_tr)
        train_probas['xgb'][val_idx] = model_xgb.predict_proba(X_val_trees)
        test_probas['xgb'] += model_xgb.predict_proba(X_test_trees) / N_FOLDS
        
        # CatBoost
        model_cat = CatBoostClassifier(**params_cat)
        model_cat.fit(X_tr_trees, y_tr)
        train_probas['cat'][val_idx] = model_cat.predict_proba(X_val_trees)
        test_probas['cat'] += model_cat.predict_proba(X_test_trees) / N_FOLDS
        
        # TabPFN
        model_tabpfn = TabPFNClassifier(device=DEVICE)
        model_tabpfn.fit(X_tr_tab.values, y_tr)
        train_probas['tabpfn'][val_idx] = model_tabpfn.predict_proba(X_val_tab.values)
        test_probas['tabpfn'] += model_tabpfn.predict_proba(X_test_tabpfn.values) / N_FOLDS

    # -------------------------------------------------------------
    # 2. Optimize Ensemble Weights
    # -------------------------------------------------------------
    print("\n🔍 Optimizing Ensemble Weights...")
    best_f1 = 0
    best_weights = None
    
    # Random search for weights
    for _ in range(2000):
        w = np.random.dirichlet(np.ones(4)) # 4 models
        
        weighted_proba = (w[0] * train_probas['lgbm'] + 
                          w[1] * train_probas['xgb'] + 
                          w[2] * train_probas['cat'] + 
                          w[3] * train_probas['tabpfn'])
        
        preds = weighted_proba.argmax(axis=1)
        f1 = f1_score(y_encoded, preds, average='macro')
        
        if f1 > best_f1:
            best_f1 = f1
            best_weights = w
            
    print(f"   Best CV F1: {best_f1:.4f}")
    print(f"   Weights: LGBM={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, CAT={best_weights[2]:.2f}, TabPFN={best_weights[3]:.2f}")
    
    # -------------------------------------------------------------
    # 3. Create Submission
    # -------------------------------------------------------------
    print("\n🔮 Generating Submission...")
    final_proba = (best_weights[0] * test_probas['lgbm'] + 
                   best_weights[1] * test_probas['xgb'] + 
                   best_weights[2] * test_probas['cat'] + 
                   best_weights[3] * test_probas['tabpfn'])
                   
    final_preds = final_proba.argmax(axis=1)
    final_labels = target_le.inverse_transform(final_preds)
    
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: final_labels})
    submission.to_csv('submission_deep_tuned.csv', index=False)
    
    print("\n✅ Saved to submission_deep_tuned.csv")
    print(f"   Prediction distribution: {pd.Series(final_labels).value_counts().to_dict()}")

if __name__ == "__main__":
    main()
