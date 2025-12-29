"""
TabPFN-2.5 Training Script (Zindi-Compliant)
Real-TabPFN-2.5: Foundation model pretrained on real data (NOT AutoML)

SETUP REQUIRED:
1. Accept license: https://huggingface.co/Prior-Labs/tabpfn_2_5/
2. Create HF token: https://huggingface.co/settings/tokens/new?tokenType=fineGrained
   (Enable "Read access to contents of all public gated repos")
3. Set environment variable: HF_TOKEN=your_token_here
   Or run: huggingface_hub.login() interactively
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings('ignore')

# Auto-detect GPU availability
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {DEVICE.upper()}")
    if DEVICE == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    DEVICE = 'cpu'
    print(f"🖥️  Device: CPU (PyTorch not available)")

# HuggingFace Authentication
try:
    import huggingface_hub
    
    # Check if token is in environment
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print("🔐 Using HF_TOKEN from environment")
        huggingface_hub.login(token=hf_token, add_to_git_credential=False)
    else:
        print("🔐 No HF_TOKEN found. Attempting interactive login...")
        print("   If this fails, set HF_TOKEN environment variable or run huggingface_hub.login() manually")
        huggingface_hub.login()
except Exception as e:
    print(f"⚠️  HuggingFace login failed: {e}")
    print("   Please ensure you have:")
    print("   1. Accepted license at https://huggingface.co/Prior-Labs/tabpfn_2_5/")
    print("   2. Created access token with gated repo permissions")
    print("   3. Set HF_TOKEN environment variable or run huggingface_hub.login()")
    raise

from tabpfn import TabPFNClassifier

SEED = 42
TARGET_COL = 'Target'
ID_COL = 'ID'

def minimal_preprocessing(df):
    """
    Minimal preprocessing - TabPFN works best with raw data
    """
    df = df.copy()
    
    # Fill missing categorical with "Unknown"
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        if col not in [ID_COL, TARGET_COL]:
            df[col] = df[col].fillna("Unknown")
    
    # Fill missing numerical with median
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    return df

def main():
    print("\n📚 Loading Data...")
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv')
    
    print("🔧 Minimal Preprocessing (preserving distribution)...")
    train_clean = minimal_preprocessing(train)
    test_clean = minimal_preprocessing(test)
    
    # Separate features and target
    X = train_clean.drop(columns=[ID_COL, TARGET_COL])
    y = train_clean[TARGET_COL]
    X_test = test_clean.drop(columns=[ID_COL])
    test_ids = test_clean[ID_COL]
    
    # Encode categorical features
    print("🔧 Label encoding categoricals...")
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    
    # Encode target
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    
    print(f"\n📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Classes: {list(target_le.classes_)}")
    
    # Validation split for local eval
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
    )
    
    # TabPFN Classifier (Real-TabPFN-2.5 default)
    print(f"\n🚀 Training Real-TabPFN-2.5 on {DEVICE.upper()}...")
    print("   (This may take a few minutes on first run due to model download)")
    
    clf = TabPFNClassifier(
        device=DEVICE,
        N_ensemble_configurations=32  # Default ensemble size
    )
    
    clf.fit(X_train.values, y_train)
    print("✅ Training Complete!")
    
    # Local validation
    print("\n📊 Validating...")
    val_preds = clf.predict(X_val.values)
    val_f1 = f1_score(y_val, val_preds, average='macro')
    print(f"   Validation F1 (Macro): {val_f1:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_val, val_preds, target_names=target_le.classes_))
    
    # Full train for final submission
    print("\n🔄 Retraining on full dataset for final predictions...")
    clf_full = TabPFNClassifier(device=DEVICE, N_ensemble_configurations=32)
    clf_full.fit(X.values, y_encoded)
    
    # Predict on test
    print("🔮 Generating test predictions...")
    preds = clf_full.predict(X_test.values)
    preds_labels = target_le.inverse_transform(preds)
    
    # Create submission
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds_labels})
    submission.to_csv('submission_tabpfn.csv', index=False)
    print("\n✅ Saved to submission_tabpfn.csv")
    print(f"   Predictions distribution: {pd.Series(preds_labels).value_counts().to_dict()}")

if __name__ == "__main__":
    main()
