import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, log_loss
from scipy.stats import randint, uniform

# Import from our new module
from financial_health_preprocessing import structured_data, preprocess_data, encode_and_scale

def optimize_hyperparameters(X, y):
    """
    Performs Randomized Search to find the best hyperparameters using intervals.
    Adapted for Kaggle resources (High iteration count).
    """
    print("\nStarting Randomized Grid Search (Intervals)...")
    
    # Define the parameter distribution (Intervals)
    param_dist = {
        'num_leaves': randint(20, 150),
        'learning_rate': uniform(0.005, 0.15), # 0.005 to ~0.155
        'n_estimators': randint(500, 3000),
        'max_depth': randint(5, 30),
        'subsample': uniform(0.6, 0.4),        # 0.6 to 1.0
        'colsample_bytree': uniform(0.6, 0.4), # 0.6 to 1.0
        'min_child_samples': randint(10, 100),
        'reg_alpha': uniform(0, 5),
        'reg_lambda': uniform(0, 5)
    }
    
    estimator = lgb.LGBMClassifier(
        objective='multiclass',
        metric='multi_logloss',
        boosting_type='gbdt',
        random_state=42,
        verbosity=-1
    )
    
    # Randomized Search
    # n_iter=50 means we test 50 random combinations from the intervals
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=50, 
        cv=3,
        scoring='f1_macro', # OPTIMIZE FOR F1-SCORE
        verbose=1,
        n_jobs=-1, # Use all cores
        random_state=42
    )
    
    search.fit(X, y)
    
    print(f"\nBest Parameters found: {search.best_params_}")
    print(f"Best F1-Macro Score: {search.best_score_:.4f}")
    
    return search.best_params_

def main():
    # 1. Load Data
    print("Loading data...")
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv')
    # sample_sub = pd.read_csv('SampleSubmission.csv') # Not strictly needed if we rebuild it
    
    # 2. Pipeline Execution
    print("\n--- Step 1: Structural Cleaning ---")
    train_struc, test_struc = structured_data(train, test)
    
    print("\n--- Step 2: Preprocessing (Imputation & Feature Eng) ---")
    train_proc, test_proc = preprocess_data(train_struc, test_struc)
    
    print("\n--- Step 3: Encoding & Scaling ---")
    X_train, y_train, X_test, le_target, scaler, train_ids, test_ids = encode_and_scale(train_proc, test_proc)
    
    print(f"\nFinal Training Shape: {X_train.shape}")
    print(f"Final Test Shape: {X_test.shape}")
    
    # 3. Hyperparameter Optimization
    # Set to True to run Grid Search
    RUN_GRID_SEARCH = True
    
    if RUN_GRID_SEARCH:
        best_params = optimize_hyperparameters(X_train, y_train)
    else:
        # Default params if skipping search
        best_params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31
        }
    
    # Ensure fixed params are present
    best_params.update({
        'objective': 'multiclass',
        'metric': 'multi_logloss', # Internal objective remains logloss for stability
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbosity': -1,
        'num_class': len(le_target.classes_)
    })

    # 4. Model Training (Stratified K-Fold) with BEST PARAMS
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    oof_preds = np.zeros((X_train.shape[0], len(le_target.classes_)))
    test_preds = np.zeros((X_test.shape[0], len(le_target.classes_)))
    
    scores_logloss = []
    scores_f1 = []
    
    print(f"\nTraining LightGBM with {N_SPLITS} folds using optimal parameters...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
        
        clf = lgb.LGBMClassifier(**best_params)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0) 
        ]
        
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',
            callbacks=callbacks
        )
        
        val_pred_proba = clf.predict_proba(X_val)
        val_preds = np.argmax(val_pred_proba, axis=1)
        oof_preds[val_idx] = val_pred_proba
        
        loss = log_loss(y_val, val_pred_proba)
        f1 = f1_score(y_val, val_preds, average='macro')
        
        scores_logloss.append(loss)
        scores_f1.append(f1)
        
        print(f"Fold {fold+1} - Log Loss: {loss:.4f} - F1 Macro: {f1:.4f}")
        
        # Predict on Test
        test_preds += clf.predict_proba(X_test) / N_SPLITS
        
    print(f"\nMean Log Loss: {np.mean(scores_logloss):.4f}")
    print(f"Mean F1 Macro: {np.mean(scores_f1):.4f}")
    
    # 5. Create Submission
    submission = pd.DataFrame(test_preds, columns=le_target.classes_)
    submission['ID'] = test_ids # Use extracted IDs to match X_test row order
    
    # Reorder columns to match SampleSubmission format
    cols = ['ID'] + list(le_target.classes_)
    submission = submission[cols]
    
    sub_filename = 'submission_lgbm_structured_optimized.csv'
    submission.to_csv(sub_filename, index=False)
    print(f"\nSubmission saved to {sub_filename}")

if __name__ == "__main__":
    main()
