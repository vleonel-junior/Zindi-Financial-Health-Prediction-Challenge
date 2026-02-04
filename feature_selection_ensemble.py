"""
🎯 FEATURE SELECTION + ULTIMATE ENSEMBLE
=========================================
Stratégie en 2 étapes:
1. Utiliser LGBM/XGBoost pour identifier les TOP features
2. Réentraîner l'ensemble (CatBoost+LGBM+XGB+TabPFN) avec ces features

Cette approche peut améliorer le F1-score en:
- Réduisant le bruit des features inutiles
- Améliorant la généralisation
- Réduisant l'overfitting
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
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)

# TabPFN check
try:
    from tabpfn import TabPFNClassifier
    import torch
    HAS_TABPFN = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ TabPFN disponible (Device: {DEVICE})")
except ImportError:
    HAS_TABPFN = False
    print("⚠️ TabPFN non disponible")


def preprocess(train_df, test_df):
    """Preprocessing minimal."""
    train = train_df.copy()
    test = test_df.copy()
    
    train_ids = train['ID'].values
    test_ids = test['ID'].values
    y = train['Target'].copy()
    
    train = train.drop(columns=['ID', 'Target'])
    test = test.drop(columns=['ID'])
    
    cat_cols = train.select_dtypes(include=['object']).columns.tolist()
    num_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Nettoyage
    for col in cat_cols:
        for apos in ["\u2019", "\u2018", "\u201B"]:
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
    
    if 'business_age_years' in train.columns:
        train['business_age_months'] = train['business_age_months'].fillna(0)
        test['business_age_months'] = test['business_age_months'].fillna(0)
        train['total_business_months'] = train['business_age_years'] * 12 + train['business_age_months']
        test['total_business_months'] = test['business_age_years'] * 12 + test['business_age_months']
        train = train.drop(columns=['business_age_years', 'business_age_months'])
        test = test.drop(columns=['business_age_years', 'business_age_months'])
    
    cat_cols = train.select_dtypes(include=['object']).columns.tolist()
    
    # Encodage
    X_train_cat = train.copy()
    X_test_cat = test.copy()
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


def get_feature_importance(X, y, method='ensemble'):
    """
    Calcule l'importance des features en utilisant LGBM + XGBoost + CatBoost.
    Retourne un DataFrame trié par importance moyenne.
    """
    print("\n" + "="*60)
    print("🔍 CALCUL DE L'IMPORTANCE DES FEATURES")
    print("="*60)
    
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Version encodée pour LGBM/XGBoost
    X_enc = X.copy()
    for col in cat_cols:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col].astype(str))
    
    # Sample weights
    class_counts = Counter(y)
    total = len(y)
    class_weights = {c: total / (len(class_counts) * count) for c, count in class_counts.items()}
    sample_weights = np.array([class_weights[yi] for yi in y])
    
    importance_df = pd.DataFrame({'feature': X.columns})
    
    # LightGBM
    print("   Training LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=50,
        class_weight='balanced', random_state=SEED, verbosity=-1
    )
    lgbm.fit(X_enc, y)
    importance_df['lgbm'] = lgbm.feature_importances_
    
    # XGBoost
    print("   Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=7,
        random_state=SEED, verbosity=0, tree_method='hist'
    )
    xgb.fit(X_enc, y, sample_weight=sample_weights)
    importance_df['xgb'] = xgb.feature_importances_
    
    # CatBoost
    print("   Training CatBoost...")
    cb = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=7,
        auto_class_weights='Balanced', random_seed=SEED, verbose=0
    )
    cb.fit(X, y, cat_features=cat_cols)
    importance_df['catboost'] = cb.feature_importances_
    
    # Normaliser chaque colonne (0-1)
    for col in ['lgbm', 'xgb', 'catboost']:
        max_val = importance_df[col].max()
        if max_val > 0:
            importance_df[col] = importance_df[col] / max_val
    
    # Moyenne pondérée
    importance_df['mean_importance'] = (
        importance_df['lgbm'] * 0.33 +
        importance_df['xgb'] * 0.33 +
        importance_df['catboost'] * 0.34
    )
    
    importance_df = importance_df.sort_values('mean_importance', ascending=False)
    
    print("\n   📊 TOP 20 FEATURES:")
    for i, row in importance_df.head(20).iterrows():
        print(f"      {row['feature']}: {row['mean_importance']:.4f}")
    
    print(f"\n   📊 BOTTOM 10 FEATURES (candidates à supprimer):")
    for i, row in importance_df.tail(10).iterrows():
        print(f"      {row['feature']}: {row['mean_importance']:.4f}")
    
    return importance_df


def select_top_features(importance_df, top_k=30, threshold=None):
    """
    Sélectionne les top K features ou celles au-dessus d'un seuil.
    """
    if threshold is not None:
        selected = importance_df[importance_df['mean_importance'] >= threshold]['feature'].tolist()
        print(f"\n   ✅ {len(selected)} features sélectionnées (seuil: {threshold})")
    else:
        selected = importance_df.head(top_k)['feature'].tolist()
        print(f"\n   ✅ {top_k} features sélectionnées (top-k)")
    
    return selected


def train_with_selected_features(data, selected_features):
    """
    Entraîne l'ensemble avec seulement les features sélectionnées.
    """
    print("\n" + "="*60)
    print(f"🏗️ ENTRAÎNEMENT AVEC {len(selected_features)} FEATURES SÉLECTIONNÉES")
    print("="*60)
    
    # Filtrer les features
    X_cat = data['X_train_cat'][selected_features]
    X_ord = data['X_train_ord'][selected_features]
    y = data['y']
    
    cat_cols = [c for c in data['cat_cols'] if c in selected_features]
    n_classes = len(data['le_target'].classes_)
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    oof_cb = np.zeros((len(y), n_classes))
    oof_lgbm = np.zeros((len(y), n_classes))
    oof_xgb = np.zeros((len(y), n_classes))
    oof_tabpfn = np.zeros((len(y), n_classes)) if HAS_TABPFN else None
    
    models_cb, models_lgbm, models_xgb = [], [], []
    models_tabpfn = [] if HAS_TABPFN else None
    
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
        
        # CatBoost
        model_cb = CatBoostClassifier(
            iterations=1500, learning_rate=0.03, depth=7, l2_leaf_reg=3,
            auto_class_weights='Balanced', random_seed=SEED, verbose=0,
            early_stopping_rounds=100
        )
        model_cb.fit(X_tr_cat, y_tr, cat_features=cat_cols,
                    eval_set=(X_val_cat, y_val), verbose=0)
        oof_cb[val_idx] = model_cb.predict_proba(X_val_cat)
        models_cb.append(model_cb)
        
        # LightGBM
        model_lgbm = LGBMClassifier(
            n_estimators=1500, learning_rate=0.03, num_leaves=50, max_depth=10,
            class_weight='balanced', random_state=SEED, verbosity=-1
        )
        model_lgbm.fit(X_tr_ord, y_tr, eval_set=[(X_val_ord, y_val)],
                      callbacks=[early_stopping(100, verbose=False)])
        oof_lgbm[val_idx] = model_lgbm.predict_proba(X_val_ord)
        models_lgbm.append(model_lgbm)
        
        # XGBoost
        model_xgb = XGBClassifier(
            n_estimators=1500, learning_rate=0.03, max_depth=7,
            random_state=SEED, verbosity=0, tree_method='hist',
            early_stopping_rounds=100
        )
        model_xgb.fit(X_tr_ord, y_tr, sample_weight=sw_tr,
                     eval_set=[(X_val_ord, y_val)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict_proba(X_val_ord)
        models_xgb.append(model_xgb)
        
        # TabPFN
        if HAS_TABPFN:
            try:
                model_tabpfn = TabPFNClassifier(device=DEVICE)
                model_tabpfn.fit(X_tr_cat.values, y_tr)
                oof_tabpfn[val_idx] = model_tabpfn.predict_proba(X_val_cat.values)
                models_tabpfn.append(model_tabpfn)
                f1_tabpfn = f1_score(y_val, oof_tabpfn[val_idx].argmax(axis=1), average='macro')
                print(f"      TabPFN: {f1_tabpfn:.4f}", end="")
            except Exception as e:
                print(f"      TabPFN Error: {e}")
                oof_tabpfn[val_idx] = (oof_cb[val_idx] + oof_lgbm[val_idx] + oof_xgb[val_idx]) / 3
                models_tabpfn.append(None)
        
        f1_cb = f1_score(y_val, oof_cb[val_idx].argmax(axis=1), average='macro')
        f1_lgbm = f1_score(y_val, oof_lgbm[val_idx].argmax(axis=1), average='macro')
        f1_xgb = f1_score(y_val, oof_xgb[val_idx].argmax(axis=1), average='macro')
        print(f"      CB: {f1_cb:.4f} | LGBM: {f1_lgbm:.4f} | XGB: {f1_xgb:.4f}")
    
    print("\n   📊 Scores OOF globaux:")
    print(f"      CatBoost: {f1_score(y, oof_cb.argmax(axis=1), average='macro'):.4f}")
    print(f"      LightGBM: {f1_score(y, oof_lgbm.argmax(axis=1), average='macro'):.4f}")
    print(f"      XGBoost:  {f1_score(y, oof_xgb.argmax(axis=1), average='macro'):.4f}")
    if HAS_TABPFN and oof_tabpfn is not None:
        print(f"      TabPFN:   {f1_score(y, oof_tabpfn.argmax(axis=1), average='macro'):.4f}")
    
    return {
        'oof_cb': oof_cb, 'oof_lgbm': oof_lgbm, 'oof_xgb': oof_xgb, 'oof_tabpfn': oof_tabpfn,
        'models_cb': models_cb, 'models_lgbm': models_lgbm, 'models_xgb': models_xgb,
        'models_tabpfn': models_tabpfn, 'selected_features': selected_features
    }


def optimize_weights(results, y):
    """Optimise les poids de l'ensemble."""
    print("\n⚖️ Optimisation des poids...")
    
    oof_cb = results['oof_cb']
    oof_lgbm = results['oof_lgbm']
    oof_xgb = results['oof_xgb']
    oof_tabpfn = results['oof_tabpfn']
    
    best_f1 = 0
    best_weights = {}
    
    if HAS_TABPFN and oof_tabpfn is not None:
        for w_cb in np.linspace(0.15, 0.45, 16):
            for w_lgbm in np.linspace(0.1, 0.35, 13):
                for w_xgb in np.linspace(0.1, 0.3, 11):
                    w_tabpfn = 1 - w_cb - w_lgbm - w_xgb
                    if w_tabpfn < 0.1 or w_tabpfn > 0.4:
                        continue
                    
                    ensemble = w_cb * oof_cb + w_lgbm * oof_lgbm + w_xgb * oof_xgb + w_tabpfn * oof_tabpfn
                    f1 = f1_score(y, ensemble.argmax(axis=1), average='macro')
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_weights = {'cb': w_cb, 'lgbm': w_lgbm, 'xgb': w_xgb, 'tabpfn': w_tabpfn}
    else:
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
    
    print(f"   Poids: {best_weights}")
    print(f"   F1 ensemble: {best_f1:.4f}")
    
    return best_weights, best_f1


def generate_predictions(data, results, weights):
    """Génère les prédictions finales."""
    print("\n🔮 Génération des prédictions...")
    
    selected_features = results['selected_features']
    X_test_cat = data['X_test_cat'][selected_features]
    X_test_ord = data['X_test_ord'][selected_features]
    n_classes = len(data['le_target'].classes_)
    
    test_cb = np.zeros((len(X_test_cat), n_classes))
    test_lgbm = np.zeros((len(X_test_cat), n_classes))
    test_xgb = np.zeros((len(X_test_cat), n_classes))
    test_tabpfn = np.zeros((len(X_test_cat), n_classes))
    
    for i in range(N_FOLDS):
        test_cb += results['models_cb'][i].predict_proba(X_test_cat) / N_FOLDS
        test_lgbm += results['models_lgbm'][i].predict_proba(X_test_ord) / N_FOLDS
        test_xgb += results['models_xgb'][i].predict_proba(X_test_ord) / N_FOLDS
        
        if HAS_TABPFN and results['models_tabpfn'][i] is not None:
            try:
                test_tabpfn += results['models_tabpfn'][i].predict_proba(X_test_cat.values) / N_FOLDS
            except:
                test_tabpfn += (test_cb + test_lgbm + test_xgb) / (3 * N_FOLDS)
    
    if 'tabpfn' in weights:
        ensemble = (weights['cb'] * test_cb + weights['lgbm'] * test_lgbm + 
                   weights['xgb'] * test_xgb + weights['tabpfn'] * test_tabpfn)
    else:
        ensemble = weights['cb'] * test_cb + weights['lgbm'] * test_lgbm + weights['xgb'] * test_xgb
    
    preds = data['le_target'].inverse_transform(ensemble.argmax(axis=1))
    print(f"   Distribution: {Counter(preds)}")
    
    return preds


def main():
    print("\n" + "🎯"*25)
    print("   FEATURE SELECTION + ULTIMATE ENSEMBLE")
    print("🎯"*25 + "\n")
    
    # Charger données
    train_df = pd.read_csv('Train.csv')
    test_df = pd.read_csv('Test.csv')
    print(f"📂 Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Preprocessing
    data = preprocess(train_df, test_df)
    print(f"   Features: {data['X_train_cat'].shape[1]}")
    
    # ÉTAPE 1: Feature importance
    importance_df = get_feature_importance(data['X_train_cat'], data['y'])
    
    # ÉTAPE 2: Sélection des features
    # Option A: Top K features
    top_k = 35  # Garder les 35 meilleures (supprimer les ~8 plus inutiles)
    
    # Option B: Seuil d'importance (décommenter pour utiliser)
    # threshold = 0.1  # Garder les features avec importance > 0.1
    # selected_features = select_top_features(importance_df, threshold=threshold)
    
    selected_features = select_top_features(importance_df, top_k=top_k)
    
    # ÉTAPE 3: Entraînement avec features sélectionnées
    results = train_with_selected_features(data, selected_features)
    
    # ÉTAPE 4: Optimisation poids
    weights, oof_f1 = optimize_weights(results, data['y'])
    
    # ÉTAPE 5: Prédictions
    preds = generate_predictions(data, results, weights)
    
    # Sauvegarde
    submission = pd.DataFrame({
        'ID': data['test_ids'],
        'Target': preds
    })
    
    filename = 'submission_feature_selection_ensemble.csv'
    submission.to_csv(filename, index=False)
    
    print(f"\n✅ Soumission: {filename}")
    print(f"   OOF F1: {oof_f1:.4f}")
    print(f"   Features utilisées: {len(selected_features)}")
    
    # Sauvegarder la liste des features importantes
    importance_df.to_csv('feature_importance.csv', index=False)
    print(f"   Feature importance sauvée: feature_importance.csv")
    
    # Classification report
    if 'tabpfn' in weights and results['oof_tabpfn'] is not None:
        ensemble_oof = (weights['cb'] * results['oof_cb'] + weights['lgbm'] * results['oof_lgbm'] +
                       weights['xgb'] * results['oof_xgb'] + weights['tabpfn'] * results['oof_tabpfn'])
    else:
        ensemble_oof = (weights['cb'] * results['oof_cb'] + weights['lgbm'] * results['oof_lgbm'] +
                       weights['xgb'] * results['oof_xgb'])
    
    print("\n📋 Classification Report:")
    print(classification_report(data['y'], ensemble_oof.argmax(axis=1), 
                               target_names=data['le_target'].classes_))


if __name__ == "__main__":
    main()
