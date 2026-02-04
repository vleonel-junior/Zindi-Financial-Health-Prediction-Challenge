"""
🚀 QUICK ENSEMBLE - Version rapide sans optimisation Optuna
============================================================
Pour tester rapidement avec des hyperparamètres par défaut optimisés.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from collections import Counter

warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)


def preprocess_quick(train_df, test_df):
    """Preprocessing minimal mais efficace."""
    print("📊 Preprocessing rapide...")
    
    train = train_df.copy()
    test = test_df.copy()
    
    train_ids = train['ID'].values
    test_ids = test['ID'].values
    y = train['Target'].copy()
    
    train = train.drop(columns=['ID', 'Target'])
    test = test.drop(columns=['ID'])
    
    cat_cols = train.select_dtypes(include=['object']).columns.tolist()
    num_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Nettoyage apostrophes
    curly_apostrophes = ["\u2019", "\u2018", "\u201B"]
    for col in cat_cols:
        for apos in curly_apostrophes:
            train[col] = train[col].astype(str).str.replace(apos, "'", regex=False)
            test[col] = test[col].astype(str).str.replace(apos, "'", regex=False)
        train[col] = train[col].str.strip().str.lower().replace('nan', np.nan)
        test[col] = test[col].str.strip().str.lower().replace('nan', np.nan)
    
    # Imputation
    for col in num_cols:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)
    
    for col in cat_cols:
        train[col] = train[col].fillna('unknown')
        test[col] = test[col].fillna('unknown')
    
    # Feature engineering
    eps = 1
    train['profit'] = train['business_turnover'] - train['business_expenses']
    test['profit'] = test['business_turnover'] - test['business_expenses']
    
    train['profit_margin'] = train['profit'] / (train['business_turnover'] + eps)
    test['profit_margin'] = test['profit'] / (test['business_turnover'] + eps)
    
    train['expense_ratio'] = train['business_expenses'] / (train['business_turnover'] + eps)
    test['expense_ratio'] = test['business_expenses'] / (test['business_turnover'] + eps)
    
    for col in ['personal_income', 'business_expenses', 'business_turnover', 'profit']:
        train[f'log_{col}'] = np.log1p(np.abs(train[col]))
        test[f'log_{col}'] = np.log1p(np.abs(test[col]))
    
    # Âge du business
    if 'business_age_years' in train.columns and 'business_age_months' in train.columns:
        train['business_age_months'] = train['business_age_months'].fillna(0)
        test['business_age_months'] = test['business_age_months'].fillna(0)
        train['total_business_months'] = train['business_age_years'] * 12 + train['business_age_months']
        test['total_business_months'] = test['business_age_years'] * 12 + test['business_age_months']
        train = train.drop(columns=['business_age_years', 'business_age_months'])
        test = test.drop(columns=['business_age_years', 'business_age_months'])
    
    # Recalculer cat_cols
    cat_cols = train.select_dtypes(include=['object']).columns.tolist()
    
    # CatBoost version (string categories)
    X_train_cat = train.copy()
    X_test_cat = test.copy()
    
    # LGBM/XGBoost version (encoded)
    X_train_ord = train.copy()
    X_test_ord = test.copy()
    
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train_ord[col], X_test_ord[col]], axis=0).astype(str)
        le.fit(combined)
        X_train_ord[col] = le.transform(X_train_ord[col].astype(str))
        X_test_ord[col] = le.transform(X_test_ord[col].astype(str))
    
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    print(f"   Shape: {X_train_cat.shape}")
    print(f"   Classes: {le_target.classes_}")
    
    return {
        'X_train_cat': X_train_cat,
        'X_test_cat': X_test_cat,
        'X_train_ord': X_train_ord,
        'X_test_ord': X_test_ord,
        'y': y_encoded,
        'le_target': le_target,
        'cat_cols': cat_cols,
        'test_ids': test_ids
    }


def train_ensemble_quick(data):
    """Entraîne l'ensemble avec des hyperparamètres par défaut optimisés."""
    print("\n🏗️ Entraînement ensemble (5-fold CV)...")
    
    X_cat = data['X_train_cat']
    X_ord = data['X_train_ord']
    y = data['y']
    cat_cols = data['cat_cols']
    n_classes = len(data['le_target'].classes_)
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    oof_cb = np.zeros((len(y), n_classes))
    oof_lgbm = np.zeros((len(y), n_classes))
    oof_xgb = np.zeros((len(y), n_classes))
    
    models_cb = []
    models_lgbm = []
    models_xgb = []
    
    # Sample weights pour XGBoost
    class_counts = Counter(y)
    total = len(y)
    class_weights = {c: total / (len(class_counts) * count) for c, count in class_counts.items()}
    sample_weights = np.array([class_weights[yi] for yi in y])
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cat, y)):
        print(f"\n   Fold {fold+1}/{N_FOLDS}")
        
        X_tr_cat, X_val_cat = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
        X_tr_ord, X_val_ord = X_ord.iloc[train_idx], X_ord.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        sw_tr = sample_weights[train_idx]
        
        # CatBoost - Hyperparamètres optimisés par défaut
        model_cb = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=3,
            border_count=128,
            random_seed=SEED,
            verbose=0,
            auto_class_weights='Balanced',
            early_stopping_rounds=100
        )
        model_cb.fit(X_tr_cat, y_tr, cat_features=cat_cols, 
                    eval_set=(X_val_cat, y_val), verbose=0)
        oof_cb[val_idx] = model_cb.predict_proba(X_val_cat)
        models_cb.append(model_cb)
        
        # LightGBM
        model_lgbm = LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=50,
            max_depth=10,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1,
            reg_lambda=1,
            random_state=SEED,
            verbosity=-1,
            class_weight='balanced',
            force_col_wise=True
        )
        model_lgbm.fit(X_tr_ord, y_tr, eval_set=[(X_val_ord, y_val)],
                      callbacks=[early_stopping(100, verbose=False)])
        oof_lgbm[val_idx] = model_lgbm.predict_proba(X_val_ord)
        models_lgbm.append(model_lgbm)
        
        # XGBoost
        model_xgb = XGBClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=7,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=1,
            reg_lambda=1,
            random_state=SEED,
            verbosity=0,
            tree_method='hist',
            early_stopping_rounds=100
        )
        model_xgb.fit(X_tr_ord, y_tr, sample_weight=sw_tr,
                     eval_set=[(X_val_ord, y_val)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict_proba(X_val_ord)
        models_xgb.append(model_xgb)
        
        # Scores
        f1_cb = f1_score(y_val, oof_cb[val_idx].argmax(axis=1), average='macro')
        f1_lgbm = f1_score(y_val, oof_lgbm[val_idx].argmax(axis=1), average='macro')
        f1_xgb = f1_score(y_val, oof_xgb[val_idx].argmax(axis=1), average='macro')
        print(f"      CB: {f1_cb:.4f} | LGBM: {f1_lgbm:.4f} | XGB: {f1_xgb:.4f}")
    
    print("\n📊 Scores OOF globaux:")
    print(f"   CatBoost: {f1_score(y, oof_cb.argmax(axis=1), average='macro'):.4f}")
    print(f"   LightGBM: {f1_score(y, oof_lgbm.argmax(axis=1), average='macro'):.4f}")
    print(f"   XGBoost:  {f1_score(y, oof_xgb.argmax(axis=1), average='macro'):.4f}")
    
    return {
        'oof_cb': oof_cb, 'oof_lgbm': oof_lgbm, 'oof_xgb': oof_xgb,
        'models_cb': models_cb, 'models_lgbm': models_lgbm, 'models_xgb': models_xgb
    }


def optimize_weights_quick(oof_cb, oof_lgbm, oof_xgb, y):
    """Recherche rapide des meilleurs poids."""
    print("\n⚖️ Optimisation des poids...")
    
    best_f1 = 0
    best_weights = (0.4, 0.3, 0.3)
    
    for w_cb in np.linspace(0.2, 0.6, 21):
        for w_lgbm in np.linspace(0.1, 0.5, 21):
            w_xgb = 1 - w_cb - w_lgbm
            if w_xgb < 0.1 or w_xgb > 0.5:
                continue
            
            ensemble = w_cb * oof_cb + w_lgbm * oof_lgbm + w_xgb * oof_xgb
            f1 = f1_score(y, ensemble.argmax(axis=1), average='macro')
            
            if f1 > best_f1:
                best_f1 = f1
                best_weights = (w_cb, w_lgbm, w_xgb)
    
    print(f"   Meilleurs poids: CB={best_weights[0]:.2f}, LGBM={best_weights[1]:.2f}, XGB={best_weights[2]:.2f}")
    print(f"   F1 ensemble: {best_f1:.4f}")
    
    return best_weights, best_f1


def generate_predictions(data, results, weights):
    """Génère les prédictions finales."""
    print("\n🔮 Génération des prédictions...")
    
    X_test_cat = data['X_test_cat']
    X_test_ord = data['X_test_ord']
    n_classes = len(data['le_target'].classes_)
    
    test_cb = np.zeros((len(X_test_cat), n_classes))
    test_lgbm = np.zeros((len(X_test_cat), n_classes))
    test_xgb = np.zeros((len(X_test_cat), n_classes))
    
    for i in range(N_FOLDS):
        test_cb += results['models_cb'][i].predict_proba(X_test_cat) / N_FOLDS
        test_lgbm += results['models_lgbm'][i].predict_proba(X_test_ord) / N_FOLDS
        test_xgb += results['models_xgb'][i].predict_proba(X_test_ord) / N_FOLDS
    
    w_cb, w_lgbm, w_xgb = weights
    ensemble = w_cb * test_cb + w_lgbm * test_lgbm + w_xgb * test_xgb
    
    preds = data['le_target'].inverse_transform(ensemble.argmax(axis=1))
    
    print(f"   Distribution: {Counter(preds)}")
    
    return preds


def main():
    print("\n" + "🚀"*20)
    print("   QUICK ENSEMBLE - Version rapide")
    print("🚀"*20 + "\n")
    
    train_df = pd.read_csv('Train.csv')
    test_df = pd.read_csv('Test.csv')
    
    print(f"📂 Train: {len(train_df)}, Test: {len(test_df)}")
    
    data = preprocess_quick(train_df, test_df)
    results = train_ensemble_quick(data)
    
    weights, oof_f1 = optimize_weights_quick(
        results['oof_cb'], results['oof_lgbm'], results['oof_xgb'], data['y']
    )
    
    preds = generate_predictions(data, results, weights)
    
    submission = pd.DataFrame({
        'ID': data['test_ids'],
        'Target': preds
    })
    
    filename = 'submission_quick_ensemble.csv'
    submission.to_csv(filename, index=False)
    
    print(f"\n✅ Soumission sauvegardée: {filename}")
    print(f"   OOF F1: {oof_f1:.4f}")
    
    # Classification report
    ensemble_oof = (weights[0] * results['oof_cb'] + 
                   weights[1] * results['oof_lgbm'] + 
                   weights[2] * results['oof_xgb'])
    
    print("\n📋 Classification Report:")
    print(classification_report(
        data['y'], ensemble_oof.argmax(axis=1),
        target_names=data['le_target'].classes_
    ))


if __name__ == "__main__":
    main()
