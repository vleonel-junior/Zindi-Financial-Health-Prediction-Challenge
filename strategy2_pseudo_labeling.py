"""
Strategy 2: Pseudo-Labeling with Confident Predictions
Uses high-confidence test predictions to augment training data
Iterative process: Train → Predict → Add confident → Retrain
Expected gain: +0.005-0.015
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
CONFIDENCE_THRESHOLD = 0.95  # Only use predictions with >95% confidence
N_ITERATIONS = 3

def preprocess(train_df, test_df):
    """Standard preprocessing"""
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

def main():
    print("\n📚 Loading Data...")
    original_train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv')
    test_ids = test[ID_COL].copy()
    
    # Preprocess
    print("🔧 Preprocessing...")
    X_train_orig, X_test, y_train_orig = preprocess(original_train, test)
    
    target_le = LabelEncoder()
    y_train_encoded = target_le.fit_transform(y_train_orig)
    
    print(f"\n📊 Original dataset: {len(y_train_orig)} train samples")
    
    # Initialize with original training data
    X_train_current = X_train_orig.copy()
    y_train_current = y_train_encoded.copy()
    X_test_current = X_test.copy()
    
    # Load best LGBM params
    try:
        with open('best_params_lgbm.json') as f:
            lgbm_params = json.load(f)
    except:
        lgbm_params = {'n_estimators': 1000, 'learning_rate': 0.05}
    lgbm_params.update({'random_state': SEED, 'verbose': -1, 'class_weight': 'balanced'})
    
    # Pseudo-labeling iterations
    for iteration in range(N_ITERATIONS):
        print(f"\n🔄 Pseudo-Labeling Iteration {iteration + 1}/{N_ITERATIONS}")
        print(f"   Current training samples: {len(y_train_current)}")
        
        # Train LGBM on current training data
        print(f"   Training LGBM...")
        lgbm = LGBMClassifier(**lgbm_params)
        lgbm.fit(X_train_current, y_train_current)
        
        # Predict on test with probabilities
        print(f"   Predicting on test set...")
        test_proba = lgbm.predict_proba(X_test_current)
        test_preds = test_proba.argmax(axis=1)
        test_confidence = test_proba.max(axis=1)
        
        # Select high-confidence predictions
        confident_mask = test_confidence >= CONFIDENCE_THRESHOLD
        n_confident = confident_mask.sum()
        
        print(f"   High-confidence predictions (>={CONFIDENCE_THRESHOLD}): {n_confident}/{len(test_confidence)} ({100*n_confident/len(test_confidence):.1f}%)")
        
        if n_confident == 0:
            print(f"   No confident predictions found. Stopping iterations.")
            break
        
        # Add confident pseudo-labels to training data
        X_pseudo = X_test_current[confident_mask]
        y_pseudo = test_preds[confident_mask]
        
        X_train_current = pd.concat([X_train_current, X_pseudo], axis=0, ignore_index=True)
        y_train_current = np.concatenate([y_train_current, y_pseudo])
        
        # Remove pseudo-labeled samples from test set for next iteration
        X_test_current = X_test_current[~confident_mask]
        
        print(f"   Added {n_confident} pseudo-labeled samples to training")
        print(f"   Remaining test samples: {len(X_test_current)}")
    
    # Final training on augmented data
    print(f"\n🔄 Final training with pseudo-labeled data...")
    print(f"   Total training samples: {len(y_train_current)} (original: {len(y_train_orig)}, added: {len(y_train_current) - len(y_train_orig)})")
    
    lgbm_final = LGBMClassifier(**lgbm_params)
    lgbm_final.fit(X_train_current, y_train_current)
    
    # Validate on original train split (20%)
    _, X_val, _, y_val = train_test_split(
        X_train_orig, y_train_encoded, test_size=0.2, random_state=SEED, stratify=y_train_encoded
    )
    
    val_preds = lgbm_final.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average='macro')
    
    print(f"\n📋 Pseudo-Labeling Validation:")
    print(f"   Validation F1 (Macro): {val_f1:.4f}")
    print(classification_report(y_val, val_preds, target_names=target_le.classes_))
    
    # Final prediction on full test set
    print("\n🔮 Generating final predictions...")
    final_preds = lgbm_final.predict(X_test)
    final_labels = target_le.inverse_transform(final_preds)
    
    # Submission
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: final_labels})
    submission.to_csv('submission_pseudo_labeling.csv', index=False)
    
    print("\n✅ Saved to submission_pseudo_labeling.csv")
    print(f"   Prediction distribution: {pd.Series(final_labels).value_counts().to_dict()}")
    print(f"\n🎯 Expected gain: +0.005-0.015 via semi-supervised learning")

if __name__ == "__main__":
    main()
