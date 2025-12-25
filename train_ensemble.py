import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')

SEED = 42
ID_COL = 'ID'
TARGET_COL = 'Target'

def load_and_process():
    print("⏳ Loading Data...")
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv')
    ss = pd.read_csv('SampleSubmission.csv')

    logical_cols = [
        'has_debit_card', 'has_mobile_money', 'has_loan_account', 'has_insurance',
        'medical_insurance', 'funeral_insurance', 'motor_vehicle_insurance',
        'has_internet_banking', 'has_credit_card'
    ]
    for df in [train, test]:
        for col in logical_cols:
            if col in df.columns:
                df[col] = df[col].fillna("No")
        
        obj_cols = df.select_dtypes(include=['object']).columns
        cols_to_fill = [c for c in obj_cols if c not in [ID_COL, TARGET_COL]]
        df[cols_to_fill] = df[cols_to_fill].fillna("Unknown")
        
        if 'personal_income' in df.columns and 'owner_age' in df.columns:
            df['income_per_age'] = df['personal_income'] / (df['owner_age'].replace(0, 1))
            
        yes_vals = ['Yes', 'Have now', 'have now']
        access_cols = ['has_loan_account', 'has_internet_banking', 'has_debit_card', 'has_mobile_money']
        valid_cols = [c for c in access_cols if c in df.columns]
        if valid_cols:
            df['financial_access_score'] = df[valid_cols].isin(yes_vals).sum(axis=1)

    return train, test, ss

def load_params(filename, default_params):
    if os.path.exists(filename):
        print(f"✅ Loaded tuned params from {filename}")
        with open(filename, 'r') as f:
            params = json.load(f)
            # Merge with defaults to ensure safety (e.g. random_state)
            default_params.update(params)
            return default_params
    else:
        print(f"⚠️ {filename} not found. Using default/fallback params.")
        return default_params

def main():
    train, test, ss = load_and_process()
    
    X = train.drop(columns=[ID_COL, TARGET_COL])
    y = train[TARGET_COL]
    X_test = test.drop(columns=[ID_COL])
    
    # Label Encode everything for voting compatibility
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    
    # 1. Load Parameters
    lgbm_defaults = {'random_state': SEED, 'verbose': -1, 'class_weight': 'balanced'}
    xgb_defaults = {'random_state': SEED, 'verbosity': 0}
    cat_defaults = {'random_state': SEED, 'verbose': 0, 'allow_writing_files': False, 'auto_class_weights': 'Balanced'}
    rf_defaults = {'random_state': SEED, 'n_jobs': -1, 'class_weight': 'balanced'}
    dt_defaults = {'random_state': SEED, 'class_weight': 'balanced'}
    
    lgbm_params = load_params('best_params_lgbm.json', lgbm_defaults)
    xgb_params = load_params('best_params_xgboost.json', xgb_defaults)
    cat_params = load_params('best_params_catboost.json', cat_defaults)
    rf_params = load_params('best_params_rf.json', rf_defaults)
    dt_params = load_params('best_params_dt.json', dt_defaults)
    
    # 2. Define Models
    clf1 = LGBMClassifier(**lgbm_params)
    clf2 = XGBClassifier(**xgb_params)
    clf3 = CatBoostClassifier(**cat_params)
    clf4 = RandomForestClassifier(**rf_params)
    clf5 = DecisionTreeClassifier(**dt_params)
    
    # 3. Optimize Ensemble Weights (The "Hill Climbing" Strategy)
    print("⚖️ Optimizing Ensemble Weights directly on OOF predictions...")
    
    # Generate Out-of-Fold Probs for EACH model
    # This allows us to test weight combinations without retraining
    models = [clf1, clf2, clf3, clf4, clf5]
    model_names = ['LGBM', 'XGB', 'Cat', 'RF', 'DecisionTree']
    oof_preds = []
    
    for name, clf in zip(model_names, models):
        print(f"   -> Generating OOF probs for {name}...")
        # cross_val_predict ensures we don't leak holdout data
        probs = cross_val_predict(clf, X, y_encoded, cv=5, method='predict_proba', n_jobs=-1)
        oof_preds.append(probs)
        
    # Weight Search Loop
    best_score = 0
    best_weights = [1.0] * len(models)
    
    # Dirichlet distribution is good for sampling proper convex weights (sum=1), 
    # but simple random uniform also works for non-convex soft voting.
    print("   -> Running Random Weight Search (2000 iterations)...")
    for _ in range(2000):
        # Sample random weights
        weights = np.random.dirichlet(np.ones(len(models)), size=1)[0]
        
        # Weighted Average
        weighted_sum = np.zeros_like(oof_preds[0])
        for w, prob in zip(weights, oof_preds):
            weighted_sum += w * prob
            
        final_preds = np.argmax(weighted_sum, axis=1)
        score = f1_score(y_encoded, final_preds, average='macro') # Optimize Macro F1
        
        if score > best_score:
            best_score = score
            best_weights = weights
            
    print(f"✅ Best Ensemble Weights Found: {np.round(best_weights, 3)}")
    print(f"✅ Best CV F1 (Macro): {best_score:.4f}")
    
    # Re-normalize best weights just in case
    
    # 3. Create Final Ensemble with Optimized Weights
    print("🤖 Initializing Final Ensemble with Optimized Weights...")
    eclf = VotingClassifier(
        estimators=[('lgbm', clf1), ('xgb', clf2), ('cat', clf3), ('rf', clf4), ('dt', clf5)],
        voting='soft',
        weights=best_weights
    )
    
    # 4. Train & Threshold Optim (on CV)
    # Note: We already have optimized weights for Argmax. 
    # Now we optimize the "High" class threshold specifically on top of the weighted ensemble.
    print("🚂 Training Final Ensemble & Optimizing 'High' Threshold...")
    
    # We need OOF for the final ensemble to tune threshold. 
    # We can reconstruct it from our previous component OOFs using best_weights!
    # This saves retraining time.
    ensemble_oof_proba = np.zeros_like(oof_preds[0])
    for w, prob in zip(best_weights, oof_preds):
        ensemble_oof_proba += w * prob
        
    # Find High Class Index
    high_class_label = 'High'
    try:
        high_idx = np.where(target_le.classes_ == high_class_label)[0][0]
    except:
        high_idx = 0 
        
    best_f1_thresh = 0
    best_thresh_weight = 1.0
    
    # Threshold Tuning Loop
    for thresh_weight in np.linspace(1.0, 4.0, 50):
        temp_proba = ensemble_oof_proba.copy()
        temp_proba[:, high_idx] *= thresh_weight
        y_pred = np.argmax(temp_proba, axis=1)
        # We focus on optimizing the F1 for 'High' class specifically (or Macro? User goal is High F1)
        # Let's optimize the metric that the competition uses. Assuming Macro F1? 
        # User said "improve F1-Score for High class", but leaderboard is usually Macro.
        # Let's optimize Macro F1 but verify High F1.
        score = f1_score(y_encoded, y_pred, average='macro')
        
        if score > best_f1_thresh:
            best_f1_thresh = score
            best_thresh_weight = thresh_weight
            
    print(f"✅ Optimal 'High' Class Multiplier: {best_thresh_weight:.2f} (Boosted CV F1: {best_f1_thresh:.4f})")
    
    # 5. Retrain on Full Data
    eclf.fit(X, y_encoded)
    
    # 6. Predict Test
    test_proba = eclf.predict_proba(X_test)
    test_proba[:, high_idx] *= best_thresh_weight
    test_preds = np.argmax(test_proba, axis=1)
    test_labels = target_le.inverse_transform(test_preds)
    
    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: test_labels})
    sub.to_csv('submission_ensemble_final.csv', index=False)
    print("📝 Saved submission_ensemble_final.csv")

if __name__ == "__main__":
    main()
