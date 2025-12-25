import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings('ignore')

SEED = 42
ID_COL = 'ID'
TARGET_COL = 'Target'
N_ITER = 20 # Number of parameter settings that are sampled.
CV_FOLDS = 5

def load_and_process():
    print("⏳ Loading Data...")
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv')
    ss = pd.read_csv('SampleSubmission.csv')

    # Logical Imputation
    logical_cols = [
        'has_debit_card', 'has_mobile_money', 'has_loan_account', 'has_insurance',
        'medical_insurance', 'funeral_insurance', 'motor_vehicle_insurance',
        'has_internet_banking', 'has_credit_card'
    ]
    for df in [train, test]:
        for col in logical_cols:
            if col in df.columns:
                df[col] = df[col].fillna("No")
        
        # Fill Unknowns
        obj_cols = df.select_dtypes(include=['object']).columns
        cols_to_fill = [c for c in obj_cols if c not in [ID_COL, TARGET_COL]]
        df[cols_to_fill] = df[cols_to_fill].fillna("Unknown")
        
        # Feature Engineering (Interaction)
        if 'personal_income' in df.columns and 'owner_age' in df.columns:
            # Avoid division by zero
            df['income_per_age'] = df['personal_income'] / (df['owner_age'].replace(0, 1))
            
        # Financial Access Score
        yes_vals = ['Yes', 'Have now', 'have now']
        access_cols = ['has_loan_account', 'has_internet_banking', 'has_debit_card', 'has_mobile_money']
        valid_cols = [c for c in access_cols if c in df.columns]
        if valid_cols:
            df['financial_access_score'] = df[valid_cols].isin(yes_vals).sum(axis=1)

    return train, test, ss

def main():
    train, test, ss = load_and_process()
    
    # Prepare X, y
    X = train.drop(columns=[ID_COL, TARGET_COL])
    y = train[TARGET_COL]
    X_test = test.drop(columns=[ID_COL])
    
    # Label Encode Categoricals
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        
    # Encode Target (High=0, Low=1, Medium=2 typically)
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    
    # Define Parameter Grid for "Deep" Tuning
    param_grid = {
        'n_estimators': [1000, 1500, 2000, 2500],
        'learning_rate': [0.01, 0.02, 0.03, 0.05],
        'num_leaves': [31, 50, 70, 100],
        'max_depth': [-1, 10, 15, 20],
        'subsample': [0.6, 0.7, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0],
        'class_weight': ['balanced', None] # Test if balanced helps or hurts
    }
    
    model = LGBMClassifier(random_state=SEED, verbose=-1, metric='multi_logloss')
    
    print(f"🔎 Starting Randomized Search for LightGBM ({N_ITER} iters)...")
    search = RandomizedSearchCV(
        model, 
        param_grid, 
        n_iter=N_ITER, 
        scoring='f1_macro', 
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED),
        verbose=1,
        random_state=SEED,
        n_jobs=-1
    )
    
    search.fit(X, y_encoded)
    
    print(f"✅ Best Params: {search.best_params_}")
    print(f"✅ Best CV F1 Score: {search.best_score_:.4f}")
    
    # Save Best Params
    with open('best_params_lgbm.json', 'w') as f:
        json.dump(search.best_params_, f)
        
    # Final Prediction
    best_clf = search.best_estimator_
    preds = best_clf.predict(X_test)
    preds_labels = target_le.inverse_transform(preds)
    
    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: preds_labels})
    sub.to_csv('submission_lgbm_tuned.csv', index=False)
    print("📝 Saved submission_lgbm_tuned.csv")

if __name__ == "__main__":
    main()
