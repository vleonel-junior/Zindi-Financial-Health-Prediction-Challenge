"""
TabPFN V3: Test avec données brutes (ZERO preprocessing)
Vérifie si TabPFN a un preprocessing interne pour les catégories
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Auto-detect GPU
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
except Exception as e:
    print(f"⚠️  HF login failed: {e}")

from tabpfn import TabPFNClassifier

SEED = 42
TARGET_COL = 'Target'
ID_COL = 'ID'

def main():
    print("\n📚 Loading Data...")
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv')
    
    # ZERO preprocessing - just fill missing
    print("🔧 ZERO Preprocessing (testing if TabPFN handles raw categories)...")
    
    X = train.drop(columns=[ID_COL, TARGET_COL])
    y = train[TARGET_COL]
    X_test = test.drop(columns=[ID_COL])
    test_ids = test[ID_COL]
    
    # Only fill missing values
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna("Unknown")
            X_test[col] = X_test[col].fillna("Unknown")
        else:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
    
    # Encode ONLY target (required for sklearn compatibility)
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    
    print(f"\n📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Categorical cols: {X.select_dtypes(include=['object']).columns.tolist()[:5]}...")
    print(f"   Numerical cols: {X.select_dtypes(include=['number']).columns.tolist()[:5]}...")
    
    # Try fitting with RAW data (categories as strings)
    print(f"\n🧪 Testing TabPFN with RAW categorical data...")
    print("   (If this fails, TabPFN needs manual encoding)")
    
    try:
        clf = TabPFNClassifier(device=DEVICE)
        
        # Validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
        )
        
        # Try fitting - will this work with strings?
        clf.fit(X_train.values, y_train)
        
        print("✅ SUCCESS! TabPFN accepted raw categorical data!")
        print("   TabPFN has internal preprocessing for categories.")
        
        # Full train
        clf_full = TabPFNClassifier(device=DEVICE)
        clf_full.fit(X.values, y_encoded)
        
        # Predict
        preds = clf_full.predict(X_test.values)
        preds_labels = target_le.inverse_transform(preds)
        
        # Submission
        submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds_labels})
        submission.to_csv('submission_tabpfn_v3_raw.csv', index=False)
        print("\n✅ Saved to submission_tabpfn_v3_raw.csv")
        
    except (ValueError, TypeError) as e:
        print(f"\n❌ FAILED: {e}")
        print("   TabPFN CANNOT handle raw categories - manual encoding required.")
        print("   Stick with V1 (Label Encoding) - Score: 0.8895")

if __name__ == "__main__":
    main()
