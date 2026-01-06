# %% --- 1. IMPORTS ET CONFIGURATION ---
import pandas as pd
import numpy as np
import warnings
import gc
import optuna 
from optuna.samplers import TPESampler

# Modèles
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression  # <--- IMPORT CORRIGÉ

# Outils Scikit-learn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

# --- CONSTANTES ---
SEED = 42
N_FOLDS = 5
MAX_TRIALS = 50
MAX_TIME = 3600 
USE_GPU = True  

# %% --- 2. FEATURE ENGINEERING ---

def process_data(train, test):
    le = LabelEncoder()
    y = le.fit_transform(train['Target'])
    
    df = pd.concat([train.drop('Target', axis=1), test], axis=0).reset_index(drop=True)
    
    # Ratios
    df['business_expenses'] = df['business_expenses'].replace(0, 1)
    df['income_to_expense'] = df['personal_income'] / df['business_expenses']
    df['margin_rate'] = (df['business_turnover'] - df['business_expenses']) / (df['business_turnover'] + 1)
    
    for col in ['personal_income', 'business_expenses', 'business_turnover']:
        df[f'log_{col}'] = np.log1p(df[col])

    # Likert Mapping
    likert_map = {'strongly disagree': 1, 'disagree': 2, 'neutral': 3, 
                  'neither agree nor disagree': 3, 'agree': 4, 'strongly agree': 5, 'nan': 3}
    
    att_cols = [c for c in df.columns if 'attitude' in c.lower() or 'perception' in c.lower()]
    if att_cols:
        for col in att_cols:
            df[col] = df[col].astype(str).str.lower().str.strip().map(likert_map).fillna(3)
        df['psych_mean'] = df[att_cols].mean(axis=1)
        df['psych_std'] = df[att_cols].std(axis=1)

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype(str).replace(['nan', 'None'], 'missing')
        
    return df.iloc[:len(train)], df.iloc[len(train):], y, le, cat_cols

# %% --- 3. TUNING (50 TRIALS / 1H) ---

def tune_models(X, y, cat_features, class_weights):
    weights_dict = dict(zip(np.unique(y), class_weights))
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_num = X.copy()
    X_num[cat_features] = oe.fit_transform(X[cat_features].astype(str))

    # CatBoost
    def obj_cb(trial):
        params = {
            'iterations': 1000,
            'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2', 1, 10),
            'task_type': 'GPU' if USE_GPU else 'CPU',
            'class_weights': list(class_weights),
            'random_seed': SEED, 'verbose': 0
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for t, v in skf.split(X, y):
            m = CatBoostClassifier(**params)
            m.fit(X.iloc[t], y.iloc[t], cat_features=cat_features, early_stopping_rounds=50)
            scores.append(f1_score(y.iloc[v], m.predict(X.iloc[v]), average='weighted'))
        return np.mean(scores)

    # XGBoost
    def obj_xgb(trial):
        params = {
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('depth', 3, 9),
            'subsample': trial.suggest_float('sub', 0.6, 1.0),
            'tree_method': 'hist', 'device': 'cuda' if USE_GPU else 'cpu',
            'random_state': SEED
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for t, v in skf.split(X_num, y):
            m = XGBClassifier(**params)
            m.fit(X_num.iloc[t], y.iloc[t], sample_weight=y.iloc[t].map(weights_dict))
            scores.append(f1_score(y.iloc[v], m.predict(X_num.iloc[v]), average='weighted'))
        return np.mean(scores)

    # LightGBM
    def obj_lgbm(trial):
        params = {
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('leaves', 20, 60),
            'device': 'gpu' if USE_GPU else 'cpu',
            'class_weight': 'balanced', 'random_state': SEED, 'verbose': -1
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for t, v in skf.split(X_num, y):
            m = LGBMClassifier(**params)
            m.fit(X_num.iloc[t], y.iloc[t])
            scores.append(f1_score(y.iloc[v], m.predict(X_num.iloc[v]), average='weighted'))
        return np.mean(scores)

    study_cb = optuna.create_study(direction='maximize'); study_cb.optimize(obj_cb, n_trials=MAX_TRIALS, timeout=MAX_TIME)
    study_xgb = optuna.create_study(direction='maximize'); study_xgb.optimize(obj_xgb, n_trials=MAX_TRIALS, timeout=MAX_TIME)
    study_lgbm = optuna.create_study(direction='maximize'); study_lgbm.optimize(obj_lgbm, n_trials=MAX_TRIALS, timeout=MAX_TIME)

    return study_cb.best_params, study_xgb.best_params, study_lgbm.best_params

# %% --- 4. EXECUTION ---

if __name__ == "__main__":
    train_raw, test_raw = pd.read_csv('Train.csv'), pd.read_csv('Test.csv')
    X, X_test, y, le, cat_cols = process_data(train_raw, test_raw)
    y_series = pd.Series(y)
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    bp_cb, bp_xgb, bp_lgbm = tune_models(X, y_series, cat_cols, class_weights)
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_stack = np.zeros((len(X), 9))
    test_stack = np.zeros((len(X_test), 9))
    
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_num, X_test_num = X.copy(), X_test.copy()
    X_num[cat_cols] = oe.fit_transform(X[cat_cols]); X_test_num[cat_cols] = oe.transform(X_test[cat_cols])

    for f, (t_idx, v_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {f+1}...")
        # CB
        m_cb = CatBoostClassifier(**bp_cb, iterations=2000, task_type='GPU' if USE_GPU else 'CPU', class_weights=list(class_weights), verbose=0)
        m_cb.fit(X.iloc[t_idx], y_series.iloc[t_idx], cat_features=cat_cols)
        oof_stack[v_idx, 0:3] = m_cb.predict_proba(X.iloc[v_idx])
        test_stack[:, 0:3] += m_cb.predict_proba(X_test) / N_FOLDS
        # XGB
        m_xgb = XGBClassifier(**bp_xgb, n_estimators=2000, tree_method='hist', device='cuda' if USE_GPU else 'cpu')
        m_xgb.fit(X_num.iloc[t_idx], y_series.iloc[t_idx])
        oof_stack[v_idx, 3:6] = m_xgb.predict_proba(X_num.iloc[v_idx])
        test_stack[:, 3:6] += m_xgb.predict_proba(X_test_num) / N_FOLDS
        # LGBM
        m_lgbm = LGBMClassifier(**bp_lgbm, n_estimators=2000, class_weight='balanced')
        m_lgbm.fit(X_num.iloc[t_idx], y_series.iloc[t_idx])
        oof_stack[v_idx, 6:9] = m_lgbm.predict_proba(X_num.iloc[v_idx])
        test_stack[:, 6:9] += m_lgbm.predict_proba(X_test_num) / N_FOLDS

    # META-LEARNER FINAL
    meta_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    meta_model.fit(oof_stack, y)
    
    pd.DataFrame({"ID": test_raw["ID"], "Target": le.inverse_transform(meta_model.predict(test_stack))}).to_csv("submission_master.csv", index=False)
    print("🚀 Terminé : submission_master.csv générée.")