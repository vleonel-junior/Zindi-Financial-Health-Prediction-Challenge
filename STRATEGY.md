# 🏆 STRATÉGIE GAGNANTE - Zindi Financial Health Challenge

## 📊 Analyse du Problème

### Dataset
- **Train**: 9,618 samples, 39 features
- **Test**: 2,405 samples
- **Taille idéale** pour TabPFN (<10K) + Gradient Boosting

### Distribution Target (DÉSÉQUILIBRE CRITIQUE!)
| Classe | Count | % |
|--------|-------|---|
| Low | 6,280 | 65.3% |
| Medium | 2,868 | 29.8% |
| **High** | **470** | **4.9%** |

⚠️ La classe **High** est très minoritaire = challenge pour le F1-score!

### Valeurs Manquantes (>20%)
26 colonnes avec >5% de manquants, dont:
- `uses_informal_lender`: 46.7%
- `uses_friends_family_savings`: 46.7%
- `motivation_make_more_money`: 44.6%
- etc.

---

## 🔧 Problèmes Identifiés dans le Preprocessing Actuel

### 1. Mapping Trop Agressif
```python
# ❌ PROBLÈME: Perte d'information
"Yes, sometimes" → "Yes"
"Never had" → "No"
"Yes, always" → "Yes"
```

**Solution**: Garder les nuances!
```python
# ✅ SOLUTION: Mapping conservateur
"Yes, sometimes" → "sometimes"
"Yes, always" → "always"
"Never had" → "never"
"Used to have..." → "used_to"
```

### 2. One-Hot Encoding Inutile
- Explose les dimensions
- Dilue l'information pour CatBoost/LGBM
- **Solution**: CatBoost gère nativement les strings!

### 3. StandardScaler Inutile
- Les modèles arbres n'ont pas besoin de normalisation
- **Solution**: Retirer le scaler

### 4. Pas de Gestion du Déséquilibre
- **Solution**: `class_weight='balanced'` + sample weights

---

## 🚀 STRATÉGIE GAGNANTE

### Architecture Ensemble Optimale

```
┌─────────────────────────────────────────────────────────┐
│                   ENSEMBLE ULTIME                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│   │  CatBoost   │  │  LightGBM   │  │   XGBoost   │    │
│   │   ~30%      │  │    ~25%     │  │    ~20%     │    │
│   │             │  │             │  │             │    │
│   │ • Cat natif │  │ • Rapide    │  │ • Robuste   │    │
│   │ • Balanced  │  │ • Balanced  │  │ • Weighted  │    │
│   └─────────────┘  └─────────────┘  └─────────────┘    │
│                                                         │
│   ┌─────────────────────────────────────────────────┐   │
│   │                    TabPFN                        │   │
│   │                     ~25%                         │   │
│   │                                                  │   │
│   │   • Zero-shot learning (pas d'overfitting!)     │   │
│   │   • Gère catégorielles nativement               │   │
│   │   • Idéal pour <10K samples                     │   │
│   └─────────────────────────────────────────────────┘   │
│                                                         │
│              ↓ Optimisation Poids ↓                     │
│                                                         │
│   ┌─────────────────────────────────────────────────┐   │
│   │        Weighted Average + Threshold Tuning       │   │
│   └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Pourquoi Cette Architecture?

| Modèle | Rôle | Force |
|--------|------|-------|
| **CatBoost** | Principal | Meilleur sur catégorielles, natif |
| **LightGBM** | Feature importance | Rapide, bon pour sélection |
| **XGBoost** | Diversité | Complémentaire, robuste |
| **TabPFN** | Zero-shot | Pas d'overfitting, diversité |

### Gestion du Déséquilibre

```python
# CatBoost
auto_class_weights='Balanced'

# LightGBM
class_weight='balanced'

# XGBoost
sample_weight = [class_weight[yi] for yi in y_train]
```

---

## 📁 Scripts Créés

### 1. `ultimate_ensemble.py` (RECOMMANDÉ)
- Ensemble complet avec TabPFN
- Optimisation automatique des poids
- 5-fold CV avec early stopping

```bash
python ultimate_ensemble.py
```

### 2. `quick_ensemble.py` (Version Rapide)
- Sans TabPFN
- Hyperparamètres pré-optimisés
- Pour tests rapides

```bash
python quick_ensemble.py
```

### 3. `winning_strategy_pipeline.py` (Version Complète)
- Optimisation Optuna des hyperparamètres
- Feature selection
- Threshold optimization

```bash
python winning_strategy_pipeline.py
```

---

## 🎯 Optimisation F1-Score

### 1. Class Weights
```python
# Calcul automatique
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
```

### 2. Threshold Tuning (Optionnel)
Pour les classes déséquilibrées, ajuster les seuils:
```python
# Seuils inversement proportionnels aux fréquences
thresholds = [proportion_class_i / sum(proportions)]
adjusted_proba = proba / thresholds
final_pred = adjusted_proba.argmax(axis=1)
```

### 3. Ensemble Weight Search
```python
# Grid search sur les poids
for w_cb in range(0.2, 0.6):
    for w_lgbm in range(0.1, 0.4):
        for w_xgb in range(0.1, 0.3):
            w_tabpfn = 1 - w_cb - w_lgbm - w_xgb
            # Calculer F1 et garder le meilleur
```

---

## 📈 Résultats Attendus

Basé sur l'analyse TabArena 2025:

| Configuration | F1 Attendu |
|--------------|-----------|
| LGBM seul | ~0.87 |
| CatBoost seul | ~0.87-0.88 |
| Ensemble CB+LGBM+XGB | ~0.88-0.89 |
| **Ensemble + TabPFN** | **~0.89-0.91** |

---

## 🔄 Workflow Recommandé

1. **Test rapide**: `python quick_ensemble.py`
2. **Si TabPFN installé**: `python ultimate_ensemble.py`
3. **Pour optimisation complète**: `python winning_strategy_pipeline.py`

---

## ⚠️ Points d'Attention

1. **TabPFN**: Nécessite `pip install tabpfn` et authentification HuggingFace
2. **GPU**: CatBoost/XGBoost/TabPFN bénéficient du GPU
3. **Mémoire**: TabPFN peut être gourmand en RAM pour les grands datasets

---

## 📝 Checklist Avant Soumission

- [ ] Vérifier la distribution des prédictions (pas trop biaisée)
- [ ] Comparer avec la distribution du train
- [ ] Vérifier le format ID,Target
- [ ] Tester plusieurs seeds pour la stabilité
