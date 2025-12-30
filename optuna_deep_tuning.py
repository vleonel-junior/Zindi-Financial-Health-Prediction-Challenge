"""
Ultra-Deep Optuna Optimization (GPU Enabled)
Optimizes LGBM, XGBoost, and CatBoost independently using 100+ trials per model.
Uses GPU acceleration for maximum speed on Kaggle.
Outputs optimal hyperparameters to JSON files for final ensembling.
"""
import pandas as pd
import numpy as np
import json
import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings
import os

warnings.filterwarnings('ignore')

# GPU Detection
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {DEVICE.upper()}")
except ImportError:
    DEVICE = 'cpu'

# Config
SEED = 42
N_TRIALS = 100  # High number of trials for deep search
ID_COL = 'ID'
TARGET_COL = 'Target'
N_FOLDS = 5

def preprocess_for_trees(train_df, test_df):
    train = train_df.drop(columns=[ID_COL, TARGET_COL])
    target = train_df[TARGET_COL]
    
    # Fill missing
    for col in train.columns:
        if train[col].dtype == 'object':
            train[col] = train[col].fillna("Unknown")
        else:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            
    # Label Encode
    cat_cols = train.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        
    # Feature Engineering
    train['income_per_age'] = train['personal_income'] / (train['owner_age'] + 1)
    
    return train, target, cat_cols

def objective_lgbm(trial, X, y, cat_cols):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'random_state': SEED,
        'verbose': -1,
        'device': 'gpu' if DEVICE == 'cuda' else 'cpu'
    }
    
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    f1_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = LGBMClassifier(**params)
        # Pass categorical_feature here? Or let it handle int encoded?
        # For simplicity and speed with deep tuning, we used int encoding above.
        # But specifying categorical_feature is better for LGBM.
        model.fit(X_train, y_train, categorical_feature=cat_cols)
        preds = model.predict(X_val)
        f1_scores.append(f1_score(y_val, preds, average='macro'))
        
    return np.mean(f1_scores)

def objective_xgb(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0), # For imbalance
        'random_state': SEED,
        'verbosity': 0,
        'tree_method': 'gpu_hist' if DEVICE == 'cuda' else 'hist'
    }
    
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    f1_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        f1_scores.append(f1_score(y_val, preds, average='macro'))
        
    return np.mean(f1_scores)

def objective_cat(trial, X, y, cat_cols):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['Balanced', 'SqrtBalanced']),
        'random_state': SEED,
        'verbose': 0,
        'task_type': 'GPU' if DEVICE == 'cuda' else 'CPU'
    }
    
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    f1_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = CatBoostClassifier(**params)
        # CatBoost handles categorical features natively if passed
        # But we pre-encoded them. CatBoost works well with int-encoded too if declared as categorical.
        # Ideally, pass 'cat_features=cat_cols' but X is all numeric now due to LabelEncoder.
        # For simplicity, treat as numeric/ordinal or pass indices if we didn't encode.
        # Given we encoded, we'll let CatBoost treat as numeric.
        model.fit(X_train, y_train) 
        preds = model.predict(X_val)
        f1_scores.append(f1_score(y_val, preds, average='macro'))
        
    return np.mean(f1_scores)

def main():
    print("🚀 Ultra-Deep Optuna Optimization Initiated...")
    
    # Load Data
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv') # Just for consistency, not used in tuning
    
    X, y, cat_cols = preprocess_for_trees(train, test)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # ---------------------------------------------------------
    # 1. Optimize LGBM
    # ---------------------------------------------------------
    print(f"\n🔬 Optimizing LGBM ({N_TRIALS} trials)...")
    study_lgbm = optuna.create_study(direction='maximize')
    study_lgbm.optimize(lambda trial: objective_lgbm(trial, X, y_encoded, cat_cols), n_trials=N_TRIALS)
    
    print(f"   ✅ Best LGBM F1: {study_lgbm.best_value:.4f}")
    print(f"   Best Params: {study_lgbm.best_params}")
    
    with open('best_params_lgbm_deep.json', 'w') as f:
        json.dump(study_lgbm.best_params, f, indent=4)
        
    # ---------------------------------------------------------
    # 2. Optimize XGBoost
    # ---------------------------------------------------------
    print(f"\n🔬 Optimizing XGBoost ({N_TRIALS} trials)...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(lambda trial: objective_xgb(trial, X, y_encoded), n_trials=N_TRIALS)
    
    print(f"   ✅ Best XGBoost F1: {study_xgb.best_value:.4f}")
    print(f"   Best Params: {study_xgb.best_params}")
    
    with open('best_params_xgboost_deep.json', 'w') as f:
        json.dump(study_xgb.best_params, f, indent=4)
        
    # ---------------------------------------------------------
    # 3. Optimize CatBoost
    # ---------------------------------------------------------
    print(f"\n🔬 Optimizing CatBoost ({N_TRIALS} trials)...")
    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(lambda trial: objective_cat(trial, X, y_encoded, cat_cols), n_trials=N_TRIALS)
    
    print(f"   ✅ Best CatBoost F1: {study_cat.best_value:.4f}")
    print(f"   Best Params: {study_cat.best_params}")
    
    with open('best_params_catboost_deep.json', 'w') as f:
        json.dump(study_cat.best_params, f, indent=4)
        
    print("\n🎉 Deep Optimization Complete!")
    print("   Parameters saved to:")
    print("   - best_params_lgbm_deep.json")
    print("   - best_params_xgboost_deep.json")
    print("   - best_params_catboost_deep.json")

if __name__ == "__main__":
    main()
