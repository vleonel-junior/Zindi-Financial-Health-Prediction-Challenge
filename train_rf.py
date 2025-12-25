import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

SEED = 42
ID_COL = 'ID'
TARGET_COL = 'Target'
N_ITER = 20 
CV_FOLDS = 5

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

def main():
    train, test, ss = load_and_process()
    
    X = train.drop(columns=[ID_COL, TARGET_COL])
    y = train[TARGET_COL]
    X_test = test.drop(columns=[ID_COL])
    
    # Label Encode for consistency (RF handles numeric best in sklearn)
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    
    # Parameter Grid for Random Forest
    param_grid = {
        'n_estimators': [500, 1000, 1500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    model = RandomForestClassifier(random_state=SEED, n_jobs=-1)
    
    print(f"🔎 Starting Randomized Search for Random Forest ({N_ITER} iters)...")
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
    
    with open('best_params_rf.json', 'w') as f:
        json.dump(search.best_params_, f)
        
    best_clf = search.best_estimator_
    preds = best_clf.predict(X_test)
    preds_labels = target_le.inverse_transform(preds)
    
    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: preds_labels})
    sub.to_csv('submission_rf_tuned.csv', index=False)
    print("📝 Saved submission_rf_tuned.csv")

if __name__ == "__main__":
    main()
