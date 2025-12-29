"""
TabPFN-2.5 V2: One-Hot Encoding for Better Categorical Handling
TabPFN Transformers can better leverage independent categorical features
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings('ignore')

# Auto-detect GPU
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {DEVICE.upper()}")
    if DEVICE == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    DEVICE = 'cpu'

# HuggingFace Auth
try:
    import huggingface_hub
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print("🔐 Using HF_TOKEN from environment")
        huggingface_hub.login(token=hf_token, add_to_git_credential=False)
    else:
        print("🔐 Attempting interactive login...")
        huggingface_hub.login()
except Exception as e:
    print(f"⚠️  HF login failed: {e}")
    raise

from tabpfn import TabPFNClassifier

SEED = 42
TARGET_COL = 'Target'
ID_COL = 'ID'

def ultra_raw_preprocessing(train_df, test_df):
    """
    Ultra-minimal preprocessing with One-Hot Encoding
    Preserves categorical independence for TabPFN's attention mechanism
    """
    # Drop IDs
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
    
    # One-Hot Encoding for categoricals (preserves independence)
    cat_cols = train.select_dtypes(include=['object']).columns
    
    if len(cat_cols) > 0:
        print(f"🔧 One-Hot Encoding {len(cat_cols)} categorical features...")
        print(f"   (This preserves category independence for TabPFN's attention)")
        
        train_encoded = pd.get_dummies(train, columns=cat_cols, drop_first=False)
        test_encoded = pd.get_dummies(test, columns=cat_cols, drop_first=False)
        
        # Align columns (test may have missing categories)
        train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)
        
        print(f"   Original features: {len(train.columns)}")
        print(f"   After One-Hot: {len(train_encoded.columns)}")
    else:
        train_encoded = train
        test_encoded = test
    
    return train_encoded, test_encoded, target

def main():
    print("\n📚 Loading Data...")
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv')
    test_ids = test[ID_COL]
    
    print("🔧 Ultra-Raw Preprocessing (One-Hot for categoricals)...")
    X, X_test, y = ultra_raw_preprocessing(train, test)
    
    # Encode target
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    
    print(f"\n📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Classes: {list(target_le.classes_)}")
    
    # Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
    )
    
    # TabPFN Classifier
    print(f"\n🚀 Training Real-TabPFN-2.5 (V2: One-Hot) on {DEVICE.upper()}...")
    
    clf = TabPFNClassifier(device=DEVICE)
    clf.fit(X_train.values, y_train)
    print("✅ Training Complete!")
    
    # Validation
    print("\n📊 Validating...")
    val_preds = clf.predict(X_val.values)
    val_f1 = f1_score(y_val, val_preds, average='macro')
    print(f"   Validation F1 (Macro): {val_f1:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_val, val_preds, target_names=target_le.classes_))
    
    # Full train
    print("\n🔄 Retraining on full dataset...")
    clf_full = TabPFNClassifier(device=DEVICE)
    clf_full.fit(X.values, y_encoded)
    
    # Predict
    print("🔮 Generating predictions...")
    preds = clf_full.predict(X_test.values)
    preds_labels = target_le.inverse_transform(preds)
    
    # Submission
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds_labels})
    submission.to_csv('submission_tabpfn_v2_onehot.csv', index=False)
    print("\n✅ Saved to submission_tabpfn_v2_onehot.csv")
    print(f"   Predictions: {pd.Series(preds_labels).value_counts().to_dict()}")

if __name__ == "__main__":
    main()
