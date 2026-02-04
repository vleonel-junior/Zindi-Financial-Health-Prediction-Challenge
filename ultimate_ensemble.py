"""
🏆 ULTIMATE ENSEMBLE - CatBoost + LightGBM + XGBoost + TabPFN
==============================================================
Stratégie optimale pour Zindi Financial Health Challenge.

TabPFN est particulièrement adapté ici car:
- Dataset ~9600 samples (< 10K, idéal pour TabPFN)
- TabPFN V3 gère les catégorielles nativement
- Zero-shot learning = pas d'overfitting

Ensemble proposé:
- CatBoost: ~30% (excellent sur catégorielles)
- LightGBM: ~25% (rapide, robuste)
- XGBoost: ~20% (complémentaire)
- TabPFN: ~25% (diversité + zero-shot)
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
import os

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
SEED = 42
N_FOLDS = 5
np.random.seed(SEED)

# Vérifier si TabPFN est disponible
try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
    print("✅ TabPFN disponible!")
    
    # Configuration GPU
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {DEVICE}")
    
except ImportError:
    HAS_TABPFN = False
    DEVICE = 'cpu'
    print("⚠️ TabPFN non installé. Utiliser: pip install tabpfn")


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_for_ensemble(train_df, test_df):
    """
    Preprocessing optimisé pour l'ensemble de modèles.
    Retourne différentes versions des données pour chaque modèle.
    """
    print("\n" + "="*60)
    print("📊 PREPROCESSING POUR ENSEMBLE")
    print("="*60)
    
    train = train_df.copy()
    test = test_df.copy()
    
    train_ids = train['ID'].values
    test_ids = test['ID'].values
    y = train['Target'].copy()
    
    train = train.drop(columns=['ID', 'Target'])
    test = test.drop(columns=['ID'])
    
    cat_cols_orig = train.select_dtypes(include=['object']).columns.tolist()
    num_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # =========================================================================
    # A. NETTOYAGE DE BASE
    # =========================================================================
    curly_apostrophes = ["\u2019", "\u2018", "\u201B"]
    
    for col in cat_cols_orig:
        for apos in curly_apostrophes:
            train[col] = train[col].astype(str).str.replace(apos, "'", regex=False)
            test[col] = test[col].astype(str).str.replace(apos, "'", regex=False)
        
        train[col] = train[col].str.strip().str.lower()
        test[col] = test[col].str.strip().str.lower()
        train[col] = train[col].replace('nan', np.nan)
        test[col] = test[col].replace('nan', np.nan)
    
    print("   ✅ Nettoyage de base effectué")
    
    # =========================================================================
    # B. IMPUTATION
    # =========================================================================
    for col in num_cols:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)
    
    for col in cat_cols_orig:
        # "unknown" pour les catégorielles manquantes
        train[col] = train[col].fillna('unknown')
        test[col] = test[col].fillna('unknown')
    
    print("   ✅ Imputation effectuée")
    
    # =========================================================================
    # C. FEATURE ENGINEERING
    # =========================================================================
    eps = 1
    
    # Ratios financiers
    train['profit'] = train['business_turnover'] - train['business_expenses']
    test['profit'] = test['business_turnover'] - test['business_expenses']
    
    train['profit_margin'] = train['profit'] / (train['business_turnover'] + eps)
    test['profit_margin'] = test['profit'] / (test['business_turnover'] + eps)
    
    train['expense_ratio'] = train['business_expenses'] / (train['business_turnover'] + eps)
    test['expense_ratio'] = test['business_expenses'] / (test['business_turnover'] + eps)
    
    train['income_to_turnover'] = train['personal_income'] / (train['business_turnover'] + eps)
    test['income_to_turnover'] = test['personal_income'] / (test['business_turnover'] + eps)
    
    # Log transforms
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
    
    print("   ✅ Feature engineering effectué")
    
    # =========================================================================
    # D. PRÉPARATION DES DIFFÉRENTES VERSIONS
    # =========================================================================
    cat_cols = train.select_dtypes(include=['object']).columns.tolist()
    
    # Version CatBoost (string categories)
    X_train_cat = train.copy()
    X_test_cat = test.copy()
    
    # Version LGBM/XGBoost (encoded)
    X_train_ord = train.copy()
    X_test_ord = test.copy()
    
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train_ord[col], X_test_ord[col]], axis=0).astype(str)
        le.fit(combined)
        X_train_ord[col] = le.transform(X_train_ord[col].astype(str))
        X_test_ord[col] = le.transform(X_test_ord[col].astype(str))
        encoders[col] = le
    
    # Version TabPFN (données RAW - TabPFN gère les catégorielles nativement)
    # TabPFN V3 peut utiliser directement les strings !
    X_train_tabpfn = train.copy()
    X_test_tabpfn = test.copy()
    
    # Target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    print(f"   ✅ Shapes: {X_train_cat.shape}")
    print(f"   ✅ Classes: {le_target.classes_}")
    
    return {
        'X_train_cat': X_train_cat,
        'X_test_cat': X_test_cat,
        'X_train_ord': X_train_ord,
        'X_test_ord': X_test_ord,
        'X_train_tabpfn': X_train_tabpfn,
        'X_test_tabpfn': X_test_tabpfn,
        'y': y_encoded,
        'le_target': le_target,
        'cat_cols': cat_cols,
        'train_ids': train_ids,
        'test_ids': test_ids
    }


# ============================================================================
# TRAINING ENSEMBLE
# ============================================================================

def train_ultimate_ensemble(data):
    """
    Entraîne l'ensemble ultime: CatBoost + LightGBM + XGBoost + TabPFN
    """
    print("\n" + "="*60)
    print("🏗️ ENTRAÎNEMENT ENSEMBLE ULTIME (5-FOLD CV)")
    print("="*60)
    
    X_cat = data['X_train_cat']
    X_ord = data['X_train_ord']
    X_tabpfn = data['X_train_tabpfn']
    y = data['y']
    cat_cols = data['cat_cols']
    n_classes = len(data['le_target'].classes_)
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    # OOF predictions
    oof_cb = np.zeros((len(y), n_classes))
    oof_lgbm = np.zeros((len(y), n_classes))
    oof_xgb = np.zeros((len(y), n_classes))
    oof_tabpfn = np.zeros((len(y), n_classes)) if HAS_TABPFN else None
    
    # Models storage
    models_cb = []
    models_lgbm = []
    models_xgb = []
    models_tabpfn = [] if HAS_TABPFN else None
    
    # Sample weights pour XGBoost
    class_counts = Counter(y)
    total = len(y)
    class_weights = {c: total / (len(class_counts) * count) for c, count in class_counts.items()}
    sample_weights = np.array([class_weights[yi] for yi in y])
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cat, y)):
        print(f"\n   📁 Fold {fold+1}/{N_FOLDS}")
        
        X_tr_cat, X_val_cat = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
        X_tr_ord, X_val_ord = X_ord.iloc[train_idx], X_ord.iloc[val_idx]
        X_tr_tabpfn, X_val_tabpfn = X_tabpfn.iloc[train_idx], X_tabpfn.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        sw_tr = sample_weights[train_idx]
        
        # -----------------------------------------------------------------
        # CatBoost
        # -----------------------------------------------------------------
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
        
        f1_cb = f1_score(y_val, oof_cb[val_idx].argmax(axis=1), average='macro')
        print(f"      CatBoost: {f1_cb:.4f}")
        
        # -----------------------------------------------------------------
        # LightGBM
        # -----------------------------------------------------------------
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
        
        f1_lgbm = f1_score(y_val, oof_lgbm[val_idx].argmax(axis=1), average='macro')
        print(f"      LightGBM: {f1_lgbm:.4f}")
        
        # -----------------------------------------------------------------
        # XGBoost
        # -----------------------------------------------------------------
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
        
        f1_xgb = f1_score(y_val, oof_xgb[val_idx].argmax(axis=1), average='macro')
        print(f"      XGBoost:  {f1_xgb:.4f}")
        
        # -----------------------------------------------------------------
        # TabPFN (si disponible)
        # -----------------------------------------------------------------
        if HAS_TABPFN:
            try:
                # TabPFN V3 avec cache pour petits datasets
                model_tabpfn = TabPFNClassifier(
                    device=DEVICE,
                    fit_mode='fit_with_cache'  # Cache pour inference rapide
                )
                
                # TabPFN peut utiliser les données RAW avec strings
                model_tabpfn.fit(X_tr_tabpfn.values, y_tr)
                oof_tabpfn[val_idx] = model_tabpfn.predict_proba(X_val_tabpfn.values)
                models_tabpfn.append(model_tabpfn)
                
                f1_tabpfn = f1_score(y_val, oof_tabpfn[val_idx].argmax(axis=1), average='macro')
                print(f"      TabPFN:   {f1_tabpfn:.4f}")
                
            except Exception as e:
                print(f"      TabPFN Error: {e}")
                # Si TabPFN échoue sur ce fold, utiliser la moyenne des autres modèles
                oof_tabpfn[val_idx] = (oof_cb[val_idx] + oof_lgbm[val_idx] + oof_xgb[val_idx]) / 3
                models_tabpfn.append(None)
    
    # Scores OOF globaux
    print("\n   📊 SCORES OOF GLOBAUX:")
    print(f"      CatBoost: {f1_score(y, oof_cb.argmax(axis=1), average='macro'):.4f}")
    print(f"      LightGBM: {f1_score(y, oof_lgbm.argmax(axis=1), average='macro'):.4f}")
    print(f"      XGBoost:  {f1_score(y, oof_xgb.argmax(axis=1), average='macro'):.4f}")
    
    if HAS_TABPFN and oof_tabpfn is not None:
        print(f"      TabPFN:   {f1_score(y, oof_tabpfn.argmax(axis=1), average='macro'):.4f}")
    
    return {
        'oof_cb': oof_cb,
        'oof_lgbm': oof_lgbm,
        'oof_xgb': oof_xgb,
        'oof_tabpfn': oof_tabpfn,
        'models_cb': models_cb,
        'models_lgbm': models_lgbm,
        'models_xgb': models_xgb,
        'models_tabpfn': models_tabpfn
    }


def optimize_ensemble_weights(results, y, has_tabpfn=False):
    """
    Recherche les poids optimaux pour l'ensemble.
    """
    print("\n" + "="*60)
    print("⚖️ OPTIMISATION DES POIDS D'ENSEMBLE")
    print("="*60)
    
    oof_cb = results['oof_cb']
    oof_lgbm = results['oof_lgbm']
    oof_xgb = results['oof_xgb']
    oof_tabpfn = results['oof_tabpfn']
    
    best_f1 = 0
    best_weights = {}
    
    if has_tabpfn and oof_tabpfn is not None:
        # 4 modèles: CatBoost, LightGBM, XGBoost, TabPFN
        print("   Recherche avec 4 modèles (CB, LGBM, XGB, TabPFN)...")
        
        for w_cb in np.linspace(0.15, 0.45, 16):
            for w_lgbm in np.linspace(0.1, 0.35, 13):
                for w_xgb in np.linspace(0.1, 0.3, 11):
                    w_tabpfn = 1 - w_cb - w_lgbm - w_xgb
                    if w_tabpfn < 0.1 or w_tabpfn > 0.4:
                        continue
                    
                    ensemble = (w_cb * oof_cb + w_lgbm * oof_lgbm + 
                               w_xgb * oof_xgb + w_tabpfn * oof_tabpfn)
                    f1 = f1_score(y, ensemble.argmax(axis=1), average='macro')
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_weights = {
                            'cb': w_cb, 'lgbm': w_lgbm, 
                            'xgb': w_xgb, 'tabpfn': w_tabpfn
                        }
    else:
        # 3 modèles seulement
        print("   Recherche avec 3 modèles (CB, LGBM, XGB)...")
        
        for w_cb in np.linspace(0.2, 0.6, 21):
            for w_lgbm in np.linspace(0.1, 0.5, 21):
                w_xgb = 1 - w_cb - w_lgbm
                if w_xgb < 0.1 or w_xgb > 0.5:
                    continue
                
                ensemble = w_cb * oof_cb + w_lgbm * oof_lgbm + w_xgb * oof_xgb
                f1 = f1_score(y, ensemble.argmax(axis=1), average='macro')
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_weights = {'cb': w_cb, 'lgbm': w_lgbm, 'xgb': w_xgb}
    
    print(f"\n   ✅ Meilleurs poids trouvés:")
    for model, weight in best_weights.items():
        print(f"      {model.upper()}: {weight:.3f}")
    print(f"   ✅ F1-score ensemble: {best_f1:.4f}")
    
    return best_weights, best_f1


def generate_final_predictions(data, results, weights):
    """
    Génère les prédictions finales sur le test set.
    """
    print("\n" + "="*60)
    print("🔮 GÉNÉRATION DES PRÉDICTIONS FINALES")
    print("="*60)
    
    X_test_cat = data['X_test_cat']
    X_test_ord = data['X_test_ord']
    X_test_tabpfn = data['X_test_tabpfn']
    n_classes = len(data['le_target'].classes_)
    
    # Moyenner les prédictions des folds
    test_cb = np.zeros((len(X_test_cat), n_classes))
    test_lgbm = np.zeros((len(X_test_cat), n_classes))
    test_xgb = np.zeros((len(X_test_cat), n_classes))
    test_tabpfn = np.zeros((len(X_test_cat), n_classes))
    
    print("   Génération des prédictions par modèle...")
    
    for i in range(N_FOLDS):
        test_cb += results['models_cb'][i].predict_proba(X_test_cat) / N_FOLDS
        test_lgbm += results['models_lgbm'][i].predict_proba(X_test_ord) / N_FOLDS
        test_xgb += results['models_xgb'][i].predict_proba(X_test_ord) / N_FOLDS
        
        if HAS_TABPFN and results['models_tabpfn'][i] is not None:
            try:
                test_tabpfn += results['models_tabpfn'][i].predict_proba(X_test_tabpfn.values) / N_FOLDS
            except:
                test_tabpfn += (test_cb + test_lgbm + test_xgb) / (3 * N_FOLDS)
    
    # Ensemble pondéré
    if 'tabpfn' in weights:
        ensemble = (weights['cb'] * test_cb + 
                   weights['lgbm'] * test_lgbm + 
                   weights['xgb'] * test_xgb + 
                   weights['tabpfn'] * test_tabpfn)
    else:
        ensemble = (weights['cb'] * test_cb + 
                   weights['lgbm'] * test_lgbm + 
                   weights['xgb'] * test_xgb)
    
    # Prédictions finales
    final_preds = data['le_target'].inverse_transform(ensemble.argmax(axis=1))
    
    # Distribution
    print(f"\n   📊 Distribution des prédictions:")
    pred_dist = Counter(final_preds)
    for label, count in sorted(pred_dist.items()):
        print(f"      {label}: {count} ({count/len(final_preds)*100:.1f}%)")
    
    return final_preds, ensemble


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "🏆"*25)
    print("   ULTIMATE ENSEMBLE - CatBoost + LightGBM + XGBoost + TabPFN")
    print("🏆"*25 + "\n")
    
    # 1. Charger les données
    print("📂 Chargement des données...")
    train_df = pd.read_csv('Train.csv')
    test_df = pd.read_csv('Test.csv')
    
    print(f"   Train: {len(train_df)} samples")
    print(f"   Test: {len(test_df)} samples")
    print(f"   TabPFN: {'✅ Disponible' if HAS_TABPFN else '❌ Non disponible'}")
    
    # 2. Preprocessing
    data = preprocess_for_ensemble(train_df, test_df)
    
    # 3. Entraînement
    results = train_ultimate_ensemble(data)
    
    # 4. Optimisation des poids
    weights, oof_f1 = optimize_ensemble_weights(
        results, data['y'], has_tabpfn=(HAS_TABPFN and results['oof_tabpfn'] is not None)
    )
    
    # 5. Prédictions finales
    final_preds, final_proba = generate_final_predictions(data, results, weights)
    
    # 6. Créer la soumission
    submission = pd.DataFrame({
        'ID': data['test_ids'],
        'Target': final_preds
    })
    
    filename = 'submission_ultimate_ensemble.csv'
    submission.to_csv(filename, index=False)
    
    print("\n" + "="*60)
    print("🎉 SOUMISSION GÉNÉRÉE!")
    print("="*60)
    print(f"   Fichier: {filename}")
    print(f"   OOF F1-score: {oof_f1:.4f}")
    
    # Classification report
    print("\n   📋 Classification Report (OOF):")
    
    if 'tabpfn' in weights and results['oof_tabpfn'] is not None:
        ensemble_oof = (weights['cb'] * results['oof_cb'] +
                       weights['lgbm'] * results['oof_lgbm'] +
                       weights['xgb'] * results['oof_xgb'] +
                       weights['tabpfn'] * results['oof_tabpfn'])
    else:
        ensemble_oof = (weights['cb'] * results['oof_cb'] +
                       weights['lgbm'] * results['oof_lgbm'] +
                       weights['xgb'] * results['oof_xgb'])
    
    print(classification_report(
        data['y'], ensemble_oof.argmax(axis=1),
        target_names=data['le_target'].classes_
    ))
    
    # Sauvegarder les poids optimaux
    print(f"\n   📊 Poids optimaux trouvés:")
    for model, weight in weights.items():
        print(f"      {model.upper()}: {weight:.3f}")


if __name__ == "__main__":
    main()
