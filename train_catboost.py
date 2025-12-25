import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
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
    
    # Label Encode (CatBoost can handle raw, but for Ensemble consistency we encode everything)
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    
    # Deep Parameter Grid for CatBoost
    param_grid = {
        'iterations': [500, 1000, 1500, 2000],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'depth': [4, 6, 8, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'random_strength': [1, 2, 5, 10],
        'bagging_temperature': [0, 1],
        'auto_class_weights': ['Balanced', 'SqrtBalanced', None]
    }
    
    # CatBoost is unique, usually RandomizedSearch is fine but specifying categorical indices is best 
    # IF we kept them as integers/strings. Since we LabelEncoded but treated as numeric for Sklearn compat, 
    # we let CatBoost treat them as numeric or categorical features index?
    # If we pass LabelEncoded ints, CatBoost treats as numeric by default unless cat_features specified.
    # For 'Deep Tuning' on Zindi, treating them as categorical is better. 
    # But since we encoded them 0..N, we should really pass cat_features indices?
    # Let's try treating them as numericals first (standard tree split on IDs) OR specify them.
    # Specifying cat_features is safer for performance.
    # Finding indices of original cat columns:
    # They are the ones we looped over.
    
    model = CatBoostClassifier(random_state=SEED, verbose=0, allow_writing_files=False)
    
    print(f"🔎 Starting Randomized Search for CatBoost ({N_ITER} iters)...")
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
    
    with open('best_params_catboost.json', 'w') as f:
        json.dump(search.best_params_, f)
        
    best_clf = search.best_estimator_
    preds = best_clf.predict(X_test)
    # CatBoost predict returns shape (N, 1) or (N,)
    preds = preds.ravel()
    preds_labels = target_le.inverse_transform(preds)
    
    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: preds_labels})
    sub.to_csv('submission_catboost_tuned.csv', index=False)
    print("📝 Saved submission_catboost_tuned.csv")

if __name__ == "__main__":
    main()
