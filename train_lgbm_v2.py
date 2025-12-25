"""
LightGBM Training with Advanced Preprocessing (V2)
"""
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import warnings
from advanced_preprocessing import advanced_preprocessing

warnings.filterwarnings('ignore')

SEED = 42
ID_COL = 'ID'
TARGET_COL = 'Target'
N_ITER = 20
CV_FOLDS = 5

def main():
    # Use advanced preprocessing
    X, y, X_test, train_ids, test_ids = advanced_preprocessing()
    
    # Encode target
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    
    # Parameter Grid
    param_grid = {
        'n_estimators': [500, 1000, 1500, 2000],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'num_leaves': [31, 50, 70, 100],
        'max_depth': [-1, 10, 20, 30],
        'min_child_samples': [10, 20, 30, 50],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5, 1.0]
    }
    
    model = LGBMClassifier(random_state=SEED, verbose=-1, class_weight='balanced')
    
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
    
    with open('best_params_lgbm_v2.json', 'w') as f:
        json.dump(search.best_params_, f)
    
    best_clf = search.best_estimator_
    preds = best_clf.predict(X_test)
    preds_labels = target_le.inverse_transform(preds)
    
    sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds_labels})
    sub.to_csv('submission_lgbm_v2.csv', index=False)
    print("📝 Saved submission_lgbm_v2.csv")

if __name__ == "__main__":
    main()
