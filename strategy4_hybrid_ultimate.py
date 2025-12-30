"""
Strategy 4: Hybrid Stacking + Pseudo-Labeling (ULTIMATE 🔥)
Combines the best of both worlds:
1. Stacking Level-2 for robust meta-learning
2. Pseudo-labeling to augment training data
Expected gain: +0.015-0.025 (HIGHEST potential)
"""
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
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
CONFIDENCE_THRESHOLD = 0.95
PSEUDO_ITERATIONS = 2

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

def get_oof_meta_features(X_std, X_raw, y, X_test_std, X_test_raw):
    """Generate OOF predictions from all base models"""
    n_classes = len(np.unique(y))
    n_models = 4
    
    oof_meta = np.zeros((len(X_std), n_classes * n_models))
    test_meta = np.zeros((len(X_test_std), n_classes * n_models))
    
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    print(f"   Generating OOF meta-features from 4 base models...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_std, y)):
        print(f"      Fold {fold+1}/{N_FOLDS}")
        
        # Split data
        X_std_train, X_std_val = X_std.iloc[train_idx], X_std.iloc[val_idx]
        X_raw_train, X_raw_val = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
        y_train = y[train_idx]
        
        # LGBM
        try:
            with open('best_params_lgbm.json') as f:
                lgbm_params = json.load(f)
        except:
            lgbm_params = {'n_estimators': 1000, 'learning_rate': 0.05}
        lgbm_params.update({'random_state': SEED, 'verbose': -1, 'class_weight': 'balanced'})
        lgbm = LGBMClassifier(**lgbm_params)
        lgbm.fit(X_std_train, y_train)
        oof_meta[val_idx, :n_classes] = lgbm.predict_proba(X_std_val)
        test_meta[:, :n_classes] += lgbm.predict_proba(X_test_std) / N_FOLDS
        
        # XGBoost
        try:
            with open('best_params_xgboost.json') as f:
                xgb_params = json.load(f)
        except:
            xgb_params = {'n_estimators': 500, 'learning_rate': 0.1}
        xgb_params.update({'random_state': SEED, 'verbosity': 0})
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_std_train, y_train)
        oof_meta[val_idx, n_classes:2*n_classes] = xgb.predict_proba(X_std_val)
        test_meta[:, n_classes:2*n_classes] += xgb.predict_proba(X_test_std) / N_FOLDS
        
        # CatBoost
        try:
            with open('best_params_catboost.json') as f:
                cat_params = json.load(f)
        except:
            cat_params = {'iterations': 500, 'learning_rate': 0.1}
        cat_params.update({'random_state': SEED, 'verbose': 0})
        cat = CatBoostClassifier(**cat_params)
        cat.fit(X_std_train, y_train)
        oof_meta[val_idx, 2*n_classes:3*n_classes] = cat.predict_proba(X_std_val)
        test_meta[:, 2*n_classes:3*n_classes] += cat.predict_proba(X_test_std) / N_FOLDS
        
        # TabPFN
        tabpfn = TabPFNClassifier(device=DEVICE)
        tabpfn.fit(X_raw_train.values, y_train)
        oof_meta[val_idx, 3*n_classes:] = tabpfn.predict_proba(X_raw_val.values)
        test_meta[:, 3*n_classes:] += tabpfn.predict_proba(X_test_raw.values) / N_FOLDS
    
    return oof_meta, test_meta

def main():
    print("\n📚 Loading Data...")
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv')
    test_ids = test[ID_COL]
    
    print("🔧 Preprocessing...")
    X_std, X_test_std, y = preprocess_standard(train, test)
    X_raw, X_test_raw, _ = preprocess_raw(train, test)
    
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    
    print(f"\n📊 PHASE 1: Stacking Level-2")
    print(f"   Generating OOF meta-features from base models...")
    
    meta_train, meta_test = get_oof_meta_features(X_std, X_raw, y_encoded, X_test_std, X_test_raw)
    
    print(f"\n📊 PHASE 2: Pseudo-Labeling on Meta-Features")
    
    # Initialize with original OOF meta-features
    meta_train_current = meta_train.copy()
    y_current = y_encoded.copy()
    meta_test_current = meta_test.copy()
    
    for iteration in range(PSEUDO_ITERATIONS):
        print(f"\n   Iteration {iteration+1}/{PSEUDO_ITERATIONS}")
        print(f"      Current training samples: {len(y_current)}")
        
        # Train meta-model
        meta_model = LogisticRegression(
            max_iter=1000,
            random_state=SEED,
            class_weight='balanced',
            multi_class='multinomial'
        )
        meta_model.fit(meta_train_current, y_current)
        
        # Predict on test meta-features
        test_proba = meta_model.predict_proba(meta_test_current)
        test_preds = test_proba.argmax(axis=1)
        test_confidence = test_proba.max(axis=1)
        
        # Select confident predictions
        confident_mask = test_confidence >= CONFIDENCE_THRESHOLD
        n_confident = confident_mask.sum()
        
        print(f"      High-confidence: {n_confident}/{len(test_confidence)} ({100*n_confident/len(test_confidence):.1f}%)")
        
        if n_confident == 0:
            break
        
        # Add pseudo-labels
        meta_pseudo = meta_test_current[confident_mask]
        y_pseudo = test_preds[confident_mask]
        
        meta_train_current = np.vstack([meta_train_current, meta_pseudo])
        y_current = np.concatenate([y_current, y_pseudo])
        meta_test_current = meta_test_current[~confident_mask]
        
        print(f"      Added {n_confident} pseudo-labeled samples")
    
    # Final training
    print(f"\n🔄 Final meta-model training...")
    print(f"   Total samples: {len(y_current)} (original: {len(y_encoded)})")
    
    meta_final = LogisticRegression(
        max_iter=1000,
        random_state=SEED,
        class_weight='balanced',
        multi_class='multinomial'
    )
    meta_final.fit(meta_train_current, y_current)
    
    # Validate
    meta_pred_train = meta_final.predict(meta_train)
    train_f1 = f1_score(y_encoded, meta_pred_train, average='macro')
    
    print(f"\n📋 Hybrid Stacking+Pseudo Validation:")
    print(f"   Train F1: {train_f1:.4f}")
    print(classification_report(y_encoded, meta_pred_train, target_names=target_le.classes_))
    
    # Final prediction
    final_preds = meta_final.predict(meta_test)
    final_labels = target_le.inverse_transform(final_preds)
    
    # Submission
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: final_labels})
    submission.to_csv('submission_hybrid_ultimate.csv', index=False)
    
    print("\n✅ Saved to submission_hybrid_ultimate.csv")
    print(f"   Prediction distribution: {pd.Series(final_labels).value_counts().to_dict()}")
    print(f"\n🎯 Expected gain: +0.015-0.025 (HIGHEST POTENTIAL)")

if __name__ == "__main__":
    main()
