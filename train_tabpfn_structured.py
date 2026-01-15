import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, log_loss, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Import ONLY cleaning steps, NOT encoding
from financial_health_preprocessing import structured_data, preprocess_data

# Suppress warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# SETUP TABPFN (GPU & AUTH)
# ------------------------------------------------------------------------------
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {DEVICE.upper()}")
except ImportError:
    DEVICE = 'cpu'

try:
    import huggingface_hub
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print("🔐 Logging in to HuggingFace...")
        huggingface_hub.login(token=hf_token, add_to_git_credential=False)
    else:
        print("⚠️  No HF_TOKEN found in env. Ensure you are logged in if using gated TabPFN 2.5.")
except Exception as e:
    print(f"⚠️  HF Login Warning: {e}")

from tabpfn import TabPFNClassifier

def main():
    print("\n📚 Loading Data...")
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv')
    
    # --------------------------------------------------------------------------
    # 1. CLEANING & PREPROCESSING (Structured + Imputed)
    # --------------------------------------------------------------------------
    print("\n--- Step 1: Structural Cleaning ---")
    train_struc, test_struc = structured_data(train, test)
    
    print("\n--- Step 2: Preprocessing (Imputation & Feature Eng) ---")
    # This fills NaNs and creates features, but keeps Strings as Strings
    train_proc, test_proc = preprocess_data(train_struc, test_struc)
    
    # --------------------------------------------------------------------------
    # 2. PREPARE FOR TABPFN
    # --------------------------------------------------------------------------
    # TabPFN supports raw strings for categories. 
    # We DO NOT One-Hot Encode or Scale.
    
    # Separate Target
    target_col = 'Target'
    if 'is_train' in train_proc.columns:
        train_proc = train_proc.drop(columns=['is_train'])
        
    X = train_proc.drop(columns=['ID', target_col], errors='ignore')
    y = train_proc[target_col]
    
    if 'is_train' in test_proc.columns:
        test_proc = test_proc.drop(columns=['is_train'])
        
    X_test = test_proc.drop(columns=['ID', target_col], errors='ignore')
    test_ids = test['ID'] # Original IDs
    
    # Encode Target (Strings -> Integers) for Scikit-Learn compatibility
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    print(f"\nTarget Classes: {le_target.classes_}")
    
    print(f"Training Shape: {X.shape}")
    print(f"Test Shape: {X_test.shape}")
    
    # --------------------------------------------------------------------------
    # 3. TRAIN TABPFN
    # --------------------------------------------------------------------------
    # Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    
    print(f"\n🚀 Training TabPFN on {DEVICE}...")
    
    # Initialize TabPFN
    # Note: No hyperparameter tuning needed!
    clf = TabPFNClassifier(device=DEVICE, random_state=42)
    
    # Fit on Train Split
    clf.fit(X_train, y_train)
    
    # Validate
    print("\n📊 Validation Metrics:")
    val_preds = clf.predict(X_val)
    val_proba = clf.predict_proba(X_val)
    
    acc = accuracy_score(y_val, val_preds)
    loss = log_loss(y_val, val_proba)
    f1 = f1_score(y_val, val_preds, average='macro')
    
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Log Loss: {loss:.4f}")
    print(f"   F1 Macro: {f1:.4f}")
    
    # --------------------------------------------------------------------------
    # 4. FULL RETRAIN & PREDICT
    # --------------------------------------------------------------------------
    print("\n🔄 Retraining on FULL dataset...")
    clf_full = TabPFNClassifier(device=DEVICE, random_state=42)
    clf_full.fit(X, y_encoded)
    
    print("🔮 Generating Predictions...")
    test_proba = clf_full.predict_proba(X_test)
    
    # Create Submission
    submission = pd.DataFrame(test_proba, columns=le_target.classes_)
    submission['ID'] = test_ids
    
    # Reorder
    cols = ['ID'] + list(le_target.classes_)
    submission = submission[cols]
    
    sub_filename = 'submission_tabpfn_structured.csv'
    submission.to_csv(sub_filename, index=False)
    print(f"\n✅ Submission saved to {sub_filename}")

if __name__ == "__main__":
    main()
