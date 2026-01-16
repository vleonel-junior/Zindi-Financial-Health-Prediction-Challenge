import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def structured_data(train_df, test_df=None):
    """
    Nettoie et standardise les données avant le prétraitement.
    - Normalise les apostrophes (curly -> straight)
    - Normalise la casse (uniformisation des valeurs catégorielles)
    - Corrige les incohérences dans les valeurs textuelles
    """
    # Combiner train et test pour un traitement cohérent
    if test_df is not None:
        # On marque temporairement pour sassurer de bien séparer plus tard si besoin, 
        # mais ici on traite le texte de manière générique
        test_df = test_df.copy()
        test_df['Target'] = np.nan 
        all_data = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    else:
        all_data = train_df.copy()

    print(f"Taille initiale (Structured Data): {all_data.shape}")
    
    _exclude_cols = ['ID', 'Target', 'country']
    _categorical_cols = all_data.select_dtypes(include=['object']).columns.tolist()
    cols_to_process = [col for col in _categorical_cols if col not in _exclude_cols]
    
    # 1. NORMALISATION DES APOSTROPHES
    curly_apostrophes = ["\u2019", "\u2018", "\u201B", "?"]
    for col in cols_to_process:
        # Vectorized replacement is faster
        for apos in curly_apostrophes:
            all_data[col] = all_data[col].astype(str).str.replace(apos, "'", regex=False)
            
    # 2. NORMALISATION DE LA CASSE + ESPACES
    for col in cols_to_process:
        # Utilisation de .str pour vectorisation
        all_data[col] = all_data[col].astype(str).str.strip().str.lower().str.capitalize()
    
    # 3. CORRECTION DES VALEURS SPÉCIFIQUES
    def assign_value_correction(s):
        if pd.isna(s) or s == 'nan':
            return np.nan
        
        s_lower = s.lower()
        
        # Valeurs positives
        if s in ["Yes", "Have now", "Yes, sometimes", "1"]:
            return "Yes"
        
        # Valeurs négatives
        if s in ["No", "Never had", "0"]:
            return "No"
        
        # Valeur "ex-utilisateur"
        if s == "Used to have but don't have now":
            return "Used to have"
        
        # Valeurs inconnues
        unknown_values = ["Don't know", "Not known", "Do not know", "Unknown", "N/a", "Refused"]
        if s in unknown_values or "don't know" in s_lower or "do not know" in s_lower:
            return np.nan
            
        return s
    
    # On applique map pour aller plus vite, ou apply
    for col in cols_to_process:
        all_data[col] = all_data[col].apply(assign_value_correction)
        
    # Séparer à nouveau Train et Test
    if test_df is not None:
        # Robust split by length (index) to ensure Test set size is preserved
        # (Target.isnull() is unsafe if Train has missing targets)
        n_train = len(train_df)
        train_structured = all_data.iloc[:n_train].copy()
        test_structured = all_data.iloc[n_train:].copy()
        
        # Remove Target from test if present (it was added as nan placeholder)
        if 'Target' in test_structured.columns:
            test_structured = test_structured.drop(columns=['Target'])
            
        return train_structured, test_structured
    else:
        return all_data

def preprocess_data(train_df, test_df):
    """
    Applique l'imputation et le feature engineering.
    CRITIQUE : Les statistiques (médianes, modes) sont calculées SUR LE TRAIN SEULEMENT.
    """
    print("-" * 30)
    print("PREPROCESSING (No Leakage Version)")
    print("-" * 30)
    
    # On copie pour ne pas modifier l'original en place
    train = train_df.copy()
    test = test_df.copy()
    
    # Pour faciliter certaines opérations communes (ex: création de variables), on concatène
    # MAIS on doit être très prudent pour l'imputation.
    target_backup = train['Target']
    train['is_train'] = 1
    test['is_train'] = 0
    test['Target'] = np.nan
    
    # ==============================================================================
    # 1. GESTION DES VALEURS MANQUANTES (Calcul sur TRAIN, Applique sur ALL)
    # ==============================================================================
    
    # --- A. Variables Numériques Financières ---
    fin_cols = ['personal_income', 'business_expenses', 'business_turnover']
    
    # Calcul des médianes par pays SUR LE TRAIN
    median_maps = {}
    for col in fin_cols:
        # On calcule la médiane par pays uniquement sur les données d'entrainement
        median_maps[col] = train.groupby('country')[col].median()
    
    # Fonction locale pour appliquer le mapping
    def fill_financial_by_country(row, col_name, medians):
        if pd.isna(row[col_name]):
            country = row['country']
            # Fallback à la médiane globale si le pays n'est pas dans le train (peu probable)
            return medians.get(country, train[col_name].median()) 
        return row[col_name]

    # Application (peut être lent avec apply, mais sûr)
    # Pour aller plus vite, on peut utiliser map et combine_first
    all_data = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    for col in fin_cols:
        # Créer une série de valeurs de remplissage basée sur le mapping
        fill_values = all_data['country'].map(median_maps[col])
        all_data[col] = all_data[col].fillna(fill_values)
        # S'il reste des NaNs (ex: nouveau pays), on remplit avec la médiane globale du train
        all_data[col] = all_data[col].fillna(train[col].median())

    # --- B. Variables Temporelles (Business Age) ---
    # Imputation valeurs fixes ou médianes train
    all_data['business_age_months'] = all_data['business_age_months'].fillna(0)
    
    median_age_years = train['business_age_years'].median()
    all_data['business_age_years'] = all_data['business_age_years'].fillna(median_age_years)
    
    all_data['total_business_age_months'] = (all_data['business_age_years'] * 12) + all_data['business_age_months']
    all_data.drop(columns=['business_age_years', 'business_age_months'], inplace=True)
    
    # --- C. Variables Catégorielles/Binaires à fort taux de manquants (>20%) ---
    # On détermine les colonnes "high missing" basées sur le TRAIN seulement pour être cohérent
    missing_ratio = train.isnull().mean()
    high_missing_cols = missing_ratio[missing_ratio > 0.2].index.tolist()
    if 'Target' in high_missing_cols: high_missing_cols.remove('Target')
    
    for col in high_missing_cols:
        if col not in all_data.columns: continue
        
        if all_data[col].dtype == 'object':
            all_data[col] = all_data[col].fillna('Unknown')
        else:
            all_data[col] = all_data[col].fillna(-1)

    # --- D. Le reste des variables ---
    remaining_cols = all_data.columns[all_data.isnull().any()].tolist()
    remaining_cols = [c for c in remaining_cols if c != 'Target']
    
    for col in remaining_cols:
        if all_data[col].dtype == 'object':
            # Mode du TRAIN
            mode_val = train[col].mode()[0]
            all_data[col] = all_data[col].fillna(mode_val)
        else:
            # Médiane du TRAIN
            median_val = train[col].median()
            all_data[col] = all_data[col].fillna(median_val)

    # ==============================================================================
    # 2. FEATURES ENGINEERING
    # ==============================================================================
    
    # Ratios Financiers
    all_data['approx_profit'] = all_data['business_turnover'] - all_data['business_expenses']
    all_data['profit_margin'] = all_data['approx_profit'] / (all_data['business_turnover'] + 1)
    all_data['expense_ratio'] = all_data['business_expenses'] / (all_data['business_turnover'] + 1)
    # Protect against div by zero or nan if personal_income was not perfectly filled (though it should be)
    all_data['personal_vs_business_income'] = all_data['personal_income'] / (all_data['business_turnover'] + 1)

    # Scores
    financial_tools = ['has_mobile_money', 'has_internet_banking', 'has_debit_card', 
                       'has_credit_card', 'has_loan_account']
    
    def check_positive(x):
        # On gère int, float, str
        if isinstance(x, str):
            x = x.lower()
            return 1 if x in ['yes', 'true', '1', 'have now'] else 0
        return 1 if x == 1 else 0

    all_data['financial_access_score'] = 0
    for col in financial_tools:
        if col in all_data.columns:
            all_data['financial_access_score'] += all_data[col].apply(check_positive)

    insurance_cols = ['motor_vehicle_insurance', 'medical_insurance', 'funeral_insurance', 'has_insurance']
    all_data['insurance_score'] = 0
    for col in insurance_cols:
        if col in all_data.columns:
            all_data['insurance_score'] += all_data[col].apply(check_positive)

    # Log Transform
    for col in ['personal_income', 'business_expenses', 'business_turnover', 'approx_profit']:
        # fillna(0) par sécurité
        all_data[f'log_{col}'] = np.log1p(np.abs(all_data[col].fillna(0)))

    # Séparation finale
    train_processed = all_data[all_data['is_train'] == 1].drop(columns=['is_train']).copy()
    test_processed = all_data[all_data['is_train'] == 0].drop(columns=['is_train', 'Target']).copy()
    
    print(f"Train processed shape: {train_processed.shape}")
    print(f"Test processed shape: {test_processed.shape}")
    
    return train_processed, test_processed

def encode_and_scale(train_df, test_df):
    """
    Encode et normalise.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Target
    le = LabelEncoder()
    y_train = le.fit_transform(train_df['Target'])
    train_df = train_df.drop(columns=['Target'])
    
    # Concat
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    all_data = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    
    # Binary Mapping
    binary_map = {'Yes': 1, 'No': 0, 'True': 1, 'False': 0, 'Unknown': -1, 'Used to have': 0.5}
    # On normalise les clés du mapping car on a déjà tout mis en capitales/minuscules ?
    # Dans structured_data on a fait capitalize(). Donc 'Yes', 'No' etc. c'est bon.
    
    binary_encoded_cols = []
    object_cols = all_data.select_dtypes(include=['object']).columns.tolist()
    
    for col in object_cols:
        # Check values
        unique_vals = set(all_data[col].dropna().unique())
        # Adaptation du check pour être robuste
        # On regarde si toutes les valeurs sont clés du map
        if unique_vals.issubset(set(binary_map.keys())):
            all_data[col] = all_data[col].map(binary_map).fillna(-1)
            binary_encoded_cols.append(col)
            
    # One Hot
    cat_cols = all_data.select_dtypes(include=['object']).columns.tolist()
    if 'ID' in cat_cols: cat_cols.remove('ID')
    
    print(f"One-Hot Encoding sur: {cat_cols}")
    all_data = pd.get_dummies(all_data, columns=cat_cols, dummy_na=False)
    
    # Scaling
    exclude = ['ID', 'is_train']
    # + binary encoded ? Généralement on scale pas les binaires 0/1 mais bon.
    # L'user voulait exclure les binaires issus du OHE.
    
    # On repère les colonnes numériques
    numeric_cols = all_data.select_dtypes(include=['number']).columns.tolist()
    cols_to_scale = []
    for col in numeric_cols:
        if col in exclude: continue
        
        # Heuristic de l'user : > 3 valeurs uniques = continu
        if all_data[col].nunique() > 3:
            cols_to_scale.append(col)
            
    print(f"Scaling sur {len(cols_to_scale)} colonnes.")
    
    # Scale
    scaler = StandardScaler()
    # FIT SUR TRAIN SEULEMENT
    train_mask = all_data['is_train'] == 1
    
    scaler.fit(all_data.loc[train_mask, cols_to_scale])
    all_data[cols_to_scale] = scaler.transform(all_data[cols_to_scale])
    
    # SANITIZE COLUMN NAMES FOR LIGHTGBM
    import re
    all_data = all_data.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))
    
    # Split
    X_train = all_data[all_data['is_train'] == 1].drop(columns=['ID', 'is_train'])
    X_test = all_data[all_data['is_train'] == 0].drop(columns=['ID', 'is_train'])
    
    # Capture IDs to ensure alignment
    train_ids = all_data[all_data['is_train'] == 1]['ID'].values
    test_ids = all_data[all_data['is_train'] == 0]['ID'].values
    
    # Vérifier alignement ID si besoin, mais ici on retourne les matrices X
    return X_train, y_train, X_test, le, scaler, train_ids, test_ids
