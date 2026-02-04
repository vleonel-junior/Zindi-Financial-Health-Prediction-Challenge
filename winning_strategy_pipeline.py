"""
🏆 WINNING STRATEGY PIPELINE - Zindi Financial Health Challenge
================================================================
Stratégie basée sur l'analyse TabArena 2025 et les spécificités du dataset.

POINTS CLÉS:
1. CatBoost en modèle principal (gestion native des catégorielles)
2. LGBM pour feature importance et sélection
3. XGBoost en complément
4. Ensemble optimisé avec recherche de poids par F1-score
5. Gestion du déséquilibre (High = 4.9% seulement!)
6. Threshold optimization pour F1-score
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
from collections import Counter

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# CONFIGURATION
# ============================================================================
SEED = 42
N_FOLDS = 5
TARGET_COL = 'Target'
ID_COL = 'ID'

np.random.seed(SEED)


# ============================================================================
# 1. PREPROCESSING OPTIMISÉ (Différent de votre associé)
# ============================================================================

def smart_preprocess(train_df, test_df):
    """
    Preprocessing OPTIMISÉ pour Gradient Boosting.
    
    DIFFÉRENCES avec le preprocessing précédent:
    1. PAS de One-Hot (CatBoost gère nativement)
    2. PAS de StandardScaler (inutile pour arbres)
    3. Mapping CONSERVATEUR (garder plus d'information)
    4. Ordinal Encoding pour LGBM/XGBoost
    """
    print("="*60)
    print("📊 PREPROCESSING OPTIMISÉ POUR GRADIENT BOOSTING")
    print("="*60)
    
    train = train_df.copy()
    test = test_df.copy()
    
    # Sauvegarder les IDs et Target
    train_ids = train[ID_COL].values
    test_ids = test[ID_COL].values
    y = train[TARGET_COL].copy()
    
    # Retirer ID et Target
    train = train.drop(columns=[ID_COL, TARGET_COL])
    test = test.drop(columns=[ID_COL])
    
    # Identifier les types de colonnes
    cat_cols = train.select_dtypes(include=['object']).columns.tolist()
    num_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"   Colonnes catégorielles: {len(cat_cols)}")
    print(f"   Colonnes numériques: {len(num_cols)}")
    
    # =========================================================================
    # A. NETTOYAGE DES APOSTROPHES ET CASSE (MINIMAL)
    # =========================================================================
    curly_apostrophes = ["\u2019", "\u2018", "\u201B"]
    
    for col in cat_cols:
        for apos in curly_apostrophes:
            train[col] = train[col].astype(str).str.replace(apos, "'", regex=False)
            test[col] = test[col].astype(str).str.replace(apos, "'", regex=False)
        
        # Normaliser la casse mais GARDER l'information originale
        train[col] = train[col].str.strip().str.lower()
        test[col] = test[col].str.strip().str.lower()
        
        # Remplacer 'nan' string par vraie valeur manquante
        train[col] = train[col].replace('nan', np.nan)
        test[col] = test[col].replace('nan', np.nan)
    
    print("   ✅ Apostrophes et casse normalisées")
    
    # =========================================================================
    # B. MAPPING CONSERVATEUR (Garder plus d'information!)
    # =========================================================================
    # IMPORTANT: Ne pas trop simplifier les catégories
    
    # Mapping pour les variables financières (has_*, uses_*, etc.)
    financial_mapping = {
        'yes': 'yes',
        'no': 'no',
        'have now': 'yes',
        'never had': 'never',  # GARDER "never" séparé de "no"!
        'used to have but don\'t have now': 'used_to',  # GARDER séparé!
        'yes, sometimes': 'sometimes',  # GARDER séparé!
        'yes, always': 'always',  # GARDER séparé!
        "don't know": np.nan,
        'don\'t know': np.nan,
        'do not know': np.nan,
        'don?t know': np.nan,
        "don't know or n/a": np.nan,
        "don?t know / doesn?t apply": np.nan,
        " do not know / n‎/a": np.nan,
    }
    
    # Colonnes binaires/ternaires spécifiques
    binary_like_cols = [col for col in cat_cols if col.startswith('has_') or 
                        col.startswith('uses_') or col.endswith('_insurance') or
                        col in ['has_insurance', 'marketing_word_of_mouth', 
                                'problem_sourcing_money', 'future_risk_theft_stock',
                                'motivation_make_more_money']]
    
    for col in binary_like_cols:
        if col in train.columns:
            train[col] = train[col].map(lambda x: financial_mapping.get(x, x) if pd.notna(x) else np.nan)
            test[col] = test[col].map(lambda x: financial_mapping.get(x, x) if pd.notna(x) else np.nan)
    
    print("   ✅ Mapping conservateur appliqué")
    
    # =========================================================================
    # C. IMPUTATION DES VALEURS MANQUANTES
    # =========================================================================
    
    # Numériques: Médiane par pays
    for col in num_cols:
        if train[col].isnull().any():
            # Médiane globale du train comme fallback
            global_median = train[col].median()
            train[col] = train[col].fillna(global_median)
            test[col] = test[col].fillna(global_median)
    
    # Catégorielles: Mode ou "unknown"
    for col in cat_cols:
        # Pour les colonnes avec beaucoup de manquants, "unknown" est informatif
        missing_ratio = train[col].isnull().mean()
        if missing_ratio > 0.3:
            train[col] = train[col].fillna('unknown')
            test[col] = test[col].fillna('unknown')
        else:
            # Mode pour les autres
            mode_val = train[col].mode()
            if len(mode_val) > 0:
                train[col] = train[col].fillna(mode_val[0])
                test[col] = test[col].fillna(mode_val[0])
            else:
                train[col] = train[col].fillna('unknown')
                test[col] = test[col].fillna('unknown')
    
    print("   ✅ Valeurs manquantes imputées")
    
    # =========================================================================
    # D. FEATURE ENGINEERING AMÉLIORÉ
    # =========================================================================
    
    # Ratios financiers (TRÈS IMPORTANTS pour la santé financière)
    eps = 1  # Pour éviter division par zéro
    
    train['profit'] = train['business_turnover'] - train['business_expenses']
    test['profit'] = test['business_turnover'] - test['business_expenses']
    
    train['profit_margin'] = train['profit'] / (train['business_turnover'] + eps)
    test['profit_margin'] = test['profit'] / (test['business_turnover'] + eps)
    
    train['expense_ratio'] = train['business_expenses'] / (train['business_turnover'] + eps)
    test['expense_ratio'] = test['business_expenses'] / (test['business_turnover'] + eps)
    
    train['income_to_turnover'] = train['personal_income'] / (train['business_turnover'] + eps)
    test['income_to_turnover'] = test['personal_income'] / (test['business_turnover'] + eps)
    
    train['income_to_expense'] = train['personal_income'] / (train['business_expenses'] + eps)
    test['income_to_expense'] = test['personal_income'] / (test['business_expenses'] + eps)
    
    # Âge total du business en mois (si les deux colonnes existent)
    if 'business_age_years' in train.columns and 'business_age_months' in train.columns:
        train['business_age_months'] = train['business_age_months'].fillna(0)
        test['business_age_months'] = test['business_age_months'].fillna(0)
        
        train['total_business_months'] = train['business_age_years'] * 12 + train['business_age_months']
        test['total_business_months'] = test['business_age_years'] * 12 + test['business_age_months']
        
        train = train.drop(columns=['business_age_years', 'business_age_months'])
        test = test.drop(columns=['business_age_years', 'business_age_months'])
    
    # Log transforms pour les variables financières (skewed)
    for col in ['personal_income', 'business_expenses', 'business_turnover', 'profit']:
        train[f'log_{col}'] = np.log1p(np.abs(train[col]))
        test[f'log_{col}'] = np.log1p(np.abs(test[col]))
    
    # Scores agrégés (utiles pour capture de patterns)
    # Score d'accès financier
    financial_cols = ['has_mobile_money', 'has_internet_banking', 'has_debit_card', 
                      'has_credit_card', 'has_loan_account']
    
    def compute_financial_score(row, cols):
        score = 0
        for col in cols:
            if col in row.index:
                val = str(row[col]).lower()
                if val in ['yes', 'always', 'sometimes']:
                    score += 1
                elif val == 'used_to':
                    score += 0.5
        return score
    
    train['financial_access_score'] = train.apply(lambda r: compute_financial_score(r, financial_cols), axis=1)
    test['financial_access_score'] = test.apply(lambda r: compute_financial_score(r, financial_cols), axis=1)
    
    # Score d'assurance
    insurance_cols = ['motor_vehicle_insurance', 'medical_insurance', 'funeral_insurance', 'has_insurance']
    train['insurance_score'] = train.apply(lambda r: compute_financial_score(r, insurance_cols), axis=1)
    test['insurance_score'] = test.apply(lambda r: compute_financial_score(r, insurance_cols), axis=1)
    
    print("   ✅ Feature engineering terminé")
    
    # =========================================================================
    # E. ENCODAGE POUR DIFFÉRENTS MODÈLES
    # =========================================================================
    
    # Recalculer les colonnes catégorielles après feature engineering
    cat_cols = train.select_dtypes(include=['object']).columns.tolist()
    
    # Pour CatBoost: Garder les catégorielles en string (natif)
    X_train_cat = train.copy()
    X_test_cat = test.copy()
    
    # Pour LGBM/XGBoost: Ordinal Encoding
    X_train_ord = train.copy()
    X_test_ord = test.copy()
    
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Fit sur train + test combinés pour éviter les nouvelles catégories
        combined = pd.concat([X_train_ord[col], X_test_ord[col]], axis=0).astype(str)
        le.fit(combined)
        X_train_ord[col] = le.transform(X_train_ord[col].astype(str))
        X_test_ord[col] = le.transform(X_test_ord[col].astype(str))
        encoders[col] = le
    
    print("   ✅ Encodage terminé")
    
    # Encoder la target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    print(f"\n   📊 Shapes finaux:")
    print(f"      X_train: {X_train_cat.shape}")
    print(f"      X_test: {X_test_cat.shape}")
    print(f"      Classes: {le_target.classes_}")
    
    return {
        'X_train_cat': X_train_cat,
        'X_test_cat': X_test_cat,
        'X_train_ord': X_train_ord,
        'X_test_ord': X_test_ord,
        'y': y_encoded,
        'le_target': le_target,
        'cat_cols': cat_cols,
        'train_ids': train_ids,
        'test_ids': test_ids,
        'encoders': encoders
    }


# ============================================================================
# 2. FEATURE SELECTION AVEC LGBM
# ============================================================================

def select_features_lgbm(X, y, cat_cols, top_k=30):
    """
    Utilise LGBM pour identifier les features les plus importantes.
    Retourne les top_k features.
    """
    print("\n" + "="*60)
    print("🔍 SÉLECTION DES FEATURES AVEC LGBM")
    print("="*60)
    
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        random_state=SEED,
        verbosity=-1,
        force_col_wise=True
    )
    
    model.fit(X, y)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n   📊 TOP 20 FEATURES:")
    for i, row in importance.head(20).iterrows():
        print(f"      {row['feature']}: {row['importance']}")
    
    # Sélectionner les top features
    top_features = importance.head(top_k)['feature'].tolist()
    
    print(f"\n   ✅ {top_k} features sélectionnées")
    
    return top_features, importance


# ============================================================================
# 3. MODÈLES AVEC OPTIMISATION HYPERPARAMÈTRES
# ============================================================================

def optimize_catboost(X, y, cat_features, n_trials=30):
    """Optimise les hyperparamètres de CatBoost avec Optuna."""
    print("\n   🔥 Optimisation CatBoost...")
    
    def objective(trial):
        params = {
            'iterations': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_seed': SEED,
            'verbose': 0,
            'auto_class_weights': 'Balanced',
            'early_stopping_rounds': 50
        }
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_val, y_val), verbose=0)
            
            preds = model.predict(X_val)
            scores.append(f1_score(y_val, preds, average='macro'))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"      Best F1: {study.best_value:.4f}")
    print(f"      Best params: {study.best_params}")
    
    return study.best_params


def optimize_lgbm(X, y, n_trials=30):
    """Optimise les hyperparamètres de LightGBM avec Optuna."""
    print("\n   🚀 Optimisation LightGBM...")
    
    def objective(trial):
        params = {
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'random_state': SEED,
            'verbosity': -1,
            'class_weight': 'balanced',
            'force_col_wise': True
        }
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = LGBMClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(50, verbose=False)])
            
            preds = model.predict(X_val)
            scores.append(f1_score(y_val, preds, average='macro'))
        
        return np.mean(scores)
    
    import lightgbm as lgb
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"      Best F1: {study.best_value:.4f}")
    print(f"      Best params: {study.best_params}")
    
    return study.best_params


def optimize_xgboost(X, y, n_trials=30):
    """Optimise les hyperparamètres de XGBoost avec Optuna."""
    print("\n   🔷 Optimisation XGBoost...")
    
    # Compute sample weights for imbalanced classes
    class_counts = Counter(y)
    total = len(y)
    class_weights = {c: total / (len(class_counts) * count) for c, count in class_counts.items()}
    sample_weights = np.array([class_weights[yi] for yi in y])
    
    def objective(trial):
        params = {
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'random_state': SEED,
            'verbosity': 0,
            'tree_method': 'hist',
            'early_stopping_rounds': 50
        }
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            sw_tr = sample_weights[train_idx]
            
            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=sw_tr,
                     eval_set=[(X_val, y_val)], verbose=False)
            
            preds = model.predict(X_val)
            scores.append(f1_score(y_val, preds, average='macro'))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"      Best F1: {study.best_value:.4f}")
    print(f"      Best params: {study.best_params}")
    
    return study.best_params


# ============================================================================
# 4. ENSEMBLE AVEC OPTIMISATION DES POIDS
# ============================================================================

def train_ensemble_cv(data, best_params_cb, best_params_lgbm, best_params_xgb):
    """
    Entraîne l'ensemble avec Cross-Validation et optimise les poids.
    """
    print("\n" + "="*60)
    print("🏗️ ENTRAÎNEMENT ENSEMBLE (5-FOLD CV)")
    print("="*60)
    
    X_cat = data['X_train_cat']
    X_ord = data['X_train_ord']
    y = data['y']
    cat_cols = data['cat_cols']
    n_classes = len(data['le_target'].classes_)
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    # Stockage OOF predictions (probabilités)
    oof_cb = np.zeros((len(y), n_classes))
    oof_lgbm = np.zeros((len(y), n_classes))
    oof_xgb = np.zeros((len(y), n_classes))
    
    # Stockage des modèles entraînés
    models_cb = []
    models_lgbm = []
    models_xgb = []
    
    # Compute sample weights for XGBoost
    class_counts = Counter(y)
    total = len(y)
    class_weights = {c: total / (len(class_counts) * count) for c, count in class_counts.items()}
    sample_weights = np.array([class_weights[yi] for yi in y])
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cat, y)):
        print(f"\n   📁 Fold {fold+1}/{N_FOLDS}")
        
        X_tr_cat, X_val_cat = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
        X_tr_ord, X_val_ord = X_ord.iloc[train_idx], X_ord.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        sw_tr = sample_weights[train_idx]
        
        # CatBoost
        cb_params = {**best_params_cb, 'iterations': 1500, 'random_seed': SEED, 
                     'verbose': 0, 'auto_class_weights': 'Balanced'}
        model_cb = CatBoostClassifier(**cb_params)
        model_cb.fit(X_tr_cat, y_tr, cat_features=cat_cols, 
                    eval_set=(X_val_cat, y_val), early_stopping_rounds=100, verbose=0)
        oof_cb[val_idx] = model_cb.predict_proba(X_val_cat)
        models_cb.append(model_cb)
        
        # LightGBM
        lgbm_params = {**best_params_lgbm, 'n_estimators': 1500, 'random_state': SEED,
                       'verbosity': -1, 'class_weight': 'balanced', 'force_col_wise': True}
        model_lgbm = LGBMClassifier(**lgbm_params)
        
        import lightgbm as lgb
        model_lgbm.fit(X_tr_ord, y_tr, eval_set=[(X_val_ord, y_val)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_lgbm[val_idx] = model_lgbm.predict_proba(X_val_ord)
        models_lgbm.append(model_lgbm)
        
        # XGBoost
        xgb_params = {**best_params_xgb, 'n_estimators': 1500, 'random_state': SEED,
                      'verbosity': 0, 'tree_method': 'hist', 'early_stopping_rounds': 100}
        model_xgb = XGBClassifier(**xgb_params)
        model_xgb.fit(X_tr_ord, y_tr, sample_weight=sw_tr,
                     eval_set=[(X_val_ord, y_val)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict_proba(X_val_ord)
        models_xgb.append(model_xgb)
        
        # Scores du fold
        f1_cb = f1_score(y_val, oof_cb[val_idx].argmax(axis=1), average='macro')
        f1_lgbm = f1_score(y_val, oof_lgbm[val_idx].argmax(axis=1), average='macro')
        f1_xgb = f1_score(y_val, oof_xgb[val_idx].argmax(axis=1), average='macro')
        
        print(f"      CatBoost F1: {f1_cb:.4f}")
        print(f"      LightGBM F1: {f1_lgbm:.4f}")
        print(f"      XGBoost  F1: {f1_xgb:.4f}")
    
    # Scores OOF globaux
    print("\n   📊 SCORES OOF GLOBAUX:")
    print(f"      CatBoost: {f1_score(y, oof_cb.argmax(axis=1), average='macro'):.4f}")
    print(f"      LightGBM: {f1_score(y, oof_lgbm.argmax(axis=1), average='macro'):.4f}")
    print(f"      XGBoost:  {f1_score(y, oof_xgb.argmax(axis=1), average='macro'):.4f}")
    
    return {
        'oof_cb': oof_cb,
        'oof_lgbm': oof_lgbm,
        'oof_xgb': oof_xgb,
        'models_cb': models_cb,
        'models_lgbm': models_lgbm,
        'models_xgb': models_xgb
    }


def optimize_ensemble_weights(oof_cb, oof_lgbm, oof_xgb, y):
    """
    Recherche les poids optimaux pour l'ensemble par grid search.
    """
    print("\n" + "="*60)
    print("⚖️ OPTIMISATION DES POIDS D'ENSEMBLE")
    print("="*60)
    
    best_f1 = 0
    best_weights = (0.4, 0.3, 0.3)  # CatBoost, LightGBM, XGBoost
    
    # Grid search sur les poids
    for w_cb in np.linspace(0.2, 0.6, 21):
        for w_lgbm in np.linspace(0.1, 0.5, 21):
            w_xgb = 1 - w_cb - w_lgbm
            if w_xgb < 0.1 or w_xgb > 0.5:
                continue
            
            ensemble_proba = w_cb * oof_cb + w_lgbm * oof_lgbm + w_xgb * oof_xgb
            preds = ensemble_proba.argmax(axis=1)
            f1 = f1_score(y, preds, average='macro')
            
            if f1 > best_f1:
                best_f1 = f1
                best_weights = (w_cb, w_lgbm, w_xgb)
    
    print(f"   ✅ Meilleurs poids trouvés:")
    print(f"      CatBoost: {best_weights[0]:.3f}")
    print(f"      LightGBM: {best_weights[1]:.3f}")
    print(f"      XGBoost:  {best_weights[2]:.3f}")
    print(f"   ✅ F1-score ensemble: {best_f1:.4f}")
    
    return best_weights, best_f1


# ============================================================================
# 5. THRESHOLD OPTIMIZATION POUR F1
# ============================================================================

def optimize_thresholds(proba, y, le_target):
    """
    Optimise les seuils de décision pour maximiser le F1-score.
    Utile quand les classes sont déséquilibrées.
    """
    print("\n" + "="*60)
    print("🎯 OPTIMISATION DES SEUILS DE DÉCISION")
    print("="*60)
    
    n_classes = proba.shape[1]
    
    # Méthode 1: Argmax simple
    preds_argmax = proba.argmax(axis=1)
    f1_argmax = f1_score(y, preds_argmax, average='macro')
    print(f"   F1 avec argmax: {f1_argmax:.4f}")
    
    # Méthode 2: Threshold ajusté par classe
    # Pour chaque classe, on cherche le seuil optimal
    best_thresholds = [0.5] * n_classes
    
    # On commence par les seuils de base (proportion inverse)
    class_counts = Counter(y)
    total = len(y)
    
    for c in range(n_classes):
        proportion = class_counts[c] / total
        # Pour les classes minoritaires, on baisse le seuil
        best_thresholds[c] = proportion
    
    # Normaliser les seuils
    sum_thresh = sum(best_thresholds)
    best_thresholds = [t / sum_thresh for t in best_thresholds]
    
    print(f"   Seuils ajustés: {[f'{t:.3f}' for t in best_thresholds]}")
    
    # Appliquer les seuils ajustés
    adjusted_proba = proba / np.array(best_thresholds)
    preds_adjusted = adjusted_proba.argmax(axis=1)
    f1_adjusted = f1_score(y, preds_adjusted, average='macro')
    print(f"   F1 avec seuils ajustés: {f1_adjusted:.4f}")
    
    # Retourner la meilleure méthode
    if f1_adjusted > f1_argmax:
        print("   ✅ Les seuils ajustés améliorent le F1!")
        return best_thresholds, 'adjusted'
    else:
        print("   ✅ Argmax donne le meilleur F1")
        return None, 'argmax'


# ============================================================================
# 6. GÉNÉRATION DES PRÉDICTIONS FINALES
# ============================================================================

def generate_final_predictions(data, ensemble_results, weights, thresholds, method):
    """
    Génère les prédictions finales sur le test set.
    """
    print("\n" + "="*60)
    print("🔮 GÉNÉRATION DES PRÉDICTIONS FINALES")
    print("="*60)
    
    X_test_cat = data['X_test_cat']
    X_test_ord = data['X_test_ord']
    n_classes = len(data['le_target'].classes_)
    
    # Moyenner les prédictions des 5 folds
    test_proba_cb = np.zeros((len(X_test_cat), n_classes))
    test_proba_lgbm = np.zeros((len(X_test_cat), n_classes))
    test_proba_xgb = np.zeros((len(X_test_cat), n_classes))
    
    for i in range(N_FOLDS):
        test_proba_cb += ensemble_results['models_cb'][i].predict_proba(X_test_cat) / N_FOLDS
        test_proba_lgbm += ensemble_results['models_lgbm'][i].predict_proba(X_test_ord) / N_FOLDS
        test_proba_xgb += ensemble_results['models_xgb'][i].predict_proba(X_test_ord) / N_FOLDS
    
    # Ensemble pondéré
    w_cb, w_lgbm, w_xgb = weights
    ensemble_proba = w_cb * test_proba_cb + w_lgbm * test_proba_lgbm + w_xgb * test_proba_xgb
    
    # Appliquer les seuils si nécessaire
    if method == 'adjusted' and thresholds is not None:
        adjusted_proba = ensemble_proba / np.array(thresholds)
        final_preds = adjusted_proba.argmax(axis=1)
    else:
        final_preds = ensemble_proba.argmax(axis=1)
    
    # Convertir en labels
    final_labels = data['le_target'].inverse_transform(final_preds)
    
    # Distribution des prédictions
    print(f"   📊 Distribution des prédictions:")
    pred_dist = Counter(final_labels)
    for label, count in sorted(pred_dist.items()):
        print(f"      {label}: {count} ({count/len(final_labels)*100:.1f}%)")
    
    return final_labels, ensemble_proba


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main():
    print("\n" + "🏆"*30)
    print("   WINNING STRATEGY PIPELINE - Zindi Financial Health Challenge")
    print("🏆"*30 + "\n")
    
    # 1. Charger les données
    print("📂 Chargement des données...")
    train_df = pd.read_csv('Train.csv')
    test_df = pd.read_csv('Test.csv')
    
    print(f"   Train: {train_df.shape[0]} samples")
    print(f"   Test: {test_df.shape[0]} samples")
    
    # 2. Preprocessing
    data = smart_preprocess(train_df, test_df)
    
    # 3. Feature Selection (optionnel - à commenter si on veut toutes les features)
    # top_features, importance = select_features_lgbm(
    #     data['X_train_ord'], data['y'], data['cat_cols'], top_k=35
    # )
    
    # 4. Optimisation des hyperparamètres
    print("\n" + "="*60)
    print("🔧 OPTIMISATION DES HYPERPARAMÈTRES")
    print("="*60)
    
    best_params_cb = optimize_catboost(
        data['X_train_cat'], data['y'], data['cat_cols'], n_trials=25
    )
    
    best_params_lgbm = optimize_lgbm(
        data['X_train_ord'], data['y'], n_trials=25
    )
    
    best_params_xgb = optimize_xgboost(
        data['X_train_ord'], data['y'], n_trials=25
    )
    
    # 5. Entraînement ensemble
    ensemble_results = train_ensemble_cv(
        data, best_params_cb, best_params_lgbm, best_params_xgb
    )
    
    # 6. Optimisation des poids
    weights, oof_f1 = optimize_ensemble_weights(
        ensemble_results['oof_cb'],
        ensemble_results['oof_lgbm'],
        ensemble_results['oof_xgb'],
        data['y']
    )
    
    # 7. Optimisation des seuils
    ensemble_oof = (weights[0] * ensemble_results['oof_cb'] + 
                    weights[1] * ensemble_results['oof_lgbm'] + 
                    weights[2] * ensemble_results['oof_xgb'])
    
    thresholds, method = optimize_thresholds(ensemble_oof, data['y'], data['le_target'])
    
    # 8. Prédictions finales
    final_labels, final_proba = generate_final_predictions(
        data, ensemble_results, weights, thresholds, method
    )
    
    # 9. Créer la soumission
    submission = pd.DataFrame({
        ID_COL: data['test_ids'],
        TARGET_COL: final_labels
    })
    
    submission_file = 'submission_winning_strategy.csv'
    submission.to_csv(submission_file, index=False)
    
    print("\n" + "="*60)
    print("🎉 SOUMISSION GÉNÉRÉE!")
    print("="*60)
    print(f"   Fichier: {submission_file}")
    print(f"   OOF F1-score: {oof_f1:.4f}")
    print(f"   Poids: CatBoost={weights[0]:.2f}, LightGBM={weights[1]:.2f}, XGBoost={weights[2]:.2f}")
    
    # Classification report sur OOF
    print("\n   📋 Classification Report (OOF):")
    oof_preds = ensemble_oof.argmax(axis=1)
    print(classification_report(
        data['y'], oof_preds, 
        target_names=data['le_target'].classes_
    ))


if __name__ == "__main__":
    main()
