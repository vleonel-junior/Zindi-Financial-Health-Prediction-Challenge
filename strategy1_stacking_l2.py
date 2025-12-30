"""
Strategy 1: Stacking Level-2 with Meta-Learner
Trains a meta-model (Logistic Regression) on out-of-fold predictions
from LGBM, TabPFN, XGBoost, and CatBoost
Expected gain: +0.010-0.020 over simple averaging
"""
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
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
N_FOLDS = 5

def preprocess_standard(train_df, test_df):
    """Standard preprocessing: Label encoding for tree models"""
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
    """RAW preprocessing for TabPFN"""
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

def get_oof_predictions(X, y, X_test, model_name='lgbm'):
    """Get out-of-fold predictions for stacking"""
    print(f"   Generating OOF predictions for {model_name.upper()}...")
    
    n_classes = len(np.unique(y))
    oof_preds = np.zeros((len(X), n_classes))
    test_preds = np.zeros((len(X_test), n_classes))
    
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if model_name == 'lgbm':
            try:
                with open('best_params_lgbm.json') as f:
                    params = json.load(f)
            except:
                params = {'n_estimators': 1000, 'learning_rate': 0.05}
            params.update({'random_state': SEED, 'verbose': -1, 'class_weight': 'balanced'})
            model = LGBMClassifier(**params)
            
        elif model_name == 'xgb':
            try:
                with open('best_params_xgboost.json') as f:
                    params = json.load(f)
            except:
                params = {'n_estimators': 500, 'learning_rate': 0.1}
            params.update({'random_state': SEED, 'verbosity': 0})
            model = XGBClassifier(**params)
            
        elif model_name == 'cat':
            try:
                with open('best_params_catboost.json') as f:
                    params = json.load(f)
            except:
                params = {'iterations': 500, 'learning_rate': 0.1}
            params.update({'random_state': SEED, 'verbose': 0})
            model = CatBoostClassifier(**params)
        
        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict_proba(X_val)
        test_preds += model.predict_proba(X_test) / N_FOLDS
    
    return oof_preds, test_preds

def main():
    print("\n📚 Loading Data...")
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv')
    test_ids = test[ID_COL]
    
    # Preprocess
    print("🔧 Preprocessing (standard for trees, raw for TabPFN)...")
    X_std, X_test_std, y = preprocess_standard(train, test)
    X_raw, X_test_raw, _ = preprocess_raw(train, test)
    
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    n_classes = len(target_le.classes_)
    
    print(f"\n📊 Dataset: {len(y)} samples, {n_classes} classes")
    print(f"   Generating OOF predictions for stacking...")
    
    # Get OOF predictions from base models
    lgbm_oof, lgbm_test = get_oof_predictions(X_std, y_encoded, X_test_std, 'lgbm')
    xgb_oof, xgb_test = get_oof_predictions(X_std, y_encoded, X_test_std, 'xgb')
    cat_oof, cat_test = get_oof_predictions(X_std, y_encoded, X_test_std, 'cat')
    
    # TabPFN OOF
    print(f"   Generating OOF predictions for TABPFN...")
    tabpfn_oof = np.zeros((len(X_raw), n_classes))
    tabpfn_test = np.zeros((len(X_test_raw), n_classes))
    
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw, y_encoded)):
        X_train, X_val = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
        y_train = y_encoded[train_idx]
        
        tabpfn = TabPFNClassifier(device=DEVICE)
        tabpfn.fit(X_train.values, y_train)
        tabpfn_oof[val_idx] = tabpfn.predict_proba(X_val.values)
        tabpfn_test += tabpfn.predict_proba(X_test_raw.values) / N_FOLDS
    
    # Stack OOF predictions as meta-features
    print("\n🔄 Training meta-learner (Logistic Regression)...")
    meta_features_train = np.hstack([lgbm_oof, xgb_oof, cat_oof, tabpfn_oof])
    meta_features_test = np.hstack([lgbm_test, xgb_test, cat_test, tabpfn_test])
    
    print(f"   Meta-features shape: {meta_features_train.shape}")
    
    # Train meta-model
    meta_model = LogisticRegression(
        max_iter=1000,
        random_state=SEED,
        class_weight='balanced',
        multi_class='multinomial',
        solver='lbfgs'
    )
    meta_model.fit(meta_features_train, y_encoded)
    
    # Validate
    meta_pred_train = meta_model.predict(meta_features_train)
    train_f1 = f1_score(y_encoded, meta_pred_train, average='macro')
    
    print(f"\n📋 Stacking Level-2 Validation:")
    print(f"   Train F1 (Macro): {train_f1:.4f}")
    print(classification_report(y_encoded, meta_pred_train, target_names=target_le.classes_))
    
    # Predict on test
    print("\n🔮 Generating final predictions...")
    meta_pred_test = meta_model.predict(meta_features_test)
    final_labels = target_le.inverse_transform(meta_pred_test)
    
    # Submission
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: final_labels})
    submission.to_csv('submission_stacking_l2.csv', index=False)
    
    print("\n✅ Saved to submission_stacking_l2.csv")
    print(f"   Prediction distribution: {pd.Series(final_labels).value_counts().to_dict()}")
    print(f"\n🎯 Expected gain: +0.010-0.020 vs simple averaging")

if __name__ == "__main__":
    main()
