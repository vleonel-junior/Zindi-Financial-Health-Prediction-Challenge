"""
Advanced Preprocessing Pipeline for Financial Health Prediction
Implements Target Encoding, Interaction Features, and Smart Binning
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

SEED = 42

def create_interaction_features(df):
    """Generate domain-specific interaction features"""
    
    # Business Maturity Index
    df['business_maturity'] = df['business_age_years'].fillna(0) * np.log1p(df['business_turnover'].fillna(1))
    
    # Financial Sophistication Score (count of modern financial tools)
    modern_tools = ['has_credit_card', 'has_internet_banking', 'has_loan_account', 'has_debit_card']
    yes_vals = ['Yes', 'Have now', 'have now']
    for col in modern_tools:
        if col in df.columns:
            df[col + '_bin'] = df[col].isin(yes_vals).astype(int)
    df['financial_sophistication'] = sum(df[col + '_bin'] for col in modern_tools if col + '_bin' in df.columns)
    
    # Income Efficiency Ratio
    df['income_efficiency'] = df['personal_income'].fillna(0) / (df['business_expenses'].fillna(1) + 1)
    
    # Profit Margin Proxy
    df['profit_margin_proxy'] = (df['business_turnover'].fillna(0) - df['business_expenses'].fillna(0)) / (df['business_turnover'].fillna(1) + 1)
    
    # Risk Perception Score (negative perceptions)
    risk_cols = ['perception_insurance_doesnt_cover_losses', 'perception_cannot_afford_insurance']
    df['risk_aversion_score'] = sum(df[col].isin(['Yes']).astype(int) for col in risk_cols if col in df.columns)
    
    # Ambition Index (positive attitudes)
    ambition_cols = ['attitude_stable_business_environment', 'attitude_more_successful_next_year']
    df['ambition_index'] = sum(df[col].isin(['Yes']).astype(int) for col in ambition_cols if col in df.columns)
    
    # Insurance Coverage Depth
    insurance_cols = ['medical_insurance', 'funeral_insurance', 'motor_vehicle_insurance']
    df['insurance_depth'] = sum(df[col].isin(yes_vals).astype(int) for col in insurance_cols if col in df.columns)
    
    # Age-Income Interaction
    df['income_per_age'] = df['personal_income'].fillna(0) / (df['owner_age'].fillna(30) + 1)
    
    # Business Scale (Log-transformed turnover)
    df['business_scale_log'] = np.log1p(df['business_turnover'].fillna(0))
    
    return df

def create_binned_features(df):
    """Create binned versions of continuous variables"""
    
    # Age bins
    df['age_bin'] = pd.cut(df['owner_age'].fillna(df['owner_age'].median()), 
                           bins=[0, 30, 45, 60, 100], 
                           labels=['young', 'mid', 'senior', 'elderly'])
    
    # Income bins (quartiles)
    income_quantiles = df['personal_income'].quantile([0.25, 0.5, 0.75]).values
    df['income_bin'] = pd.cut(df['personal_income'].fillna(0), 
                              bins=[-np.inf, income_quantiles[0], income_quantiles[1], income_quantiles[2], np.inf],
                              labels=['low', 'med_low', 'med_high', 'high'])
    
    # Business age bins
    df['business_age_bin'] = pd.cut(df['business_age_years'].fillna(0), 
                                     bins=[-1, 1, 5, 10, 100], 
                                     labels=['startup', 'growing', 'established', 'mature'])
    
    return df

def target_encode_feature(train_df, test_df, col, target_col, smoothing=10):
    """
    Target encoding with K-Fold cross-validation to prevent overfitting
    """
    # Global mean
    global_mean = train_df[target_col].map({'Low': 0, 'Medium': 1, 'High': 2}).mean()
    
    # Encoding map for train (with CV)
    train_encoded = np.zeros(len(train_df))
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for train_idx, val_idx in kf.split(train_df):
        # Calculate mean target per category on train fold
        target_numeric = train_df.iloc[train_idx][target_col].map({'Low': 0, 'Medium': 1, 'High': 2})
        category_means = train_df.iloc[train_idx].groupby(col)[target_col].apply(
            lambda x: x.map({'Low': 0, 'Medium': 1, 'High': 2}).mean()
        )
        category_counts = train_df.iloc[train_idx][col].value_counts()
        
        # Smoothing
        smoothed_means = {}
        for cat in category_means.index:
            count = category_counts.get(cat, 0)
            smoothed_means[cat] = (category_means[cat] * count + global_mean * smoothing) / (count + smoothing)
        
        # Encode validation fold
        train_encoded[val_idx] = train_df.iloc[val_idx][col].map(smoothed_means).fillna(global_mean)
    
    # Encoding map for test (using full train)
    target_numeric = train_df[target_col].map({'Low': 0, 'Medium': 1, 'High': 2})
    category_means = train_df.groupby(col)[target_col].apply(
        lambda x: x.map({'Low': 0, 'Medium': 1, 'High': 2}).mean()
    )
    category_counts = train_df[col].value_counts()
    
    smoothed_means = {}
    for cat in category_means.index:
        count = category_counts.get(cat, 0)
        smoothed_means[cat] = (category_means[cat] * count + global_mean * smoothing) / (count + smoothing)
    
    test_encoded = test_df[col].map(smoothed_means).fillna(global_mean)
    
    return train_encoded, test_encoded

def advanced_preprocessing(train_path='Train.csv', test_path='Test.csv', use_target_encoding=True):
    """
    Full advanced preprocessing pipeline
    Returns: X_train, y_train, X_test, train_ids, test_ids
    """
    print("🔧 Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Save IDs and target
    train_ids = train['ID'].copy()
    test_ids = test['ID'].copy()
    target = train['Target'].copy()
    
    # Drop IDs and target from features
    train = train.drop(columns=['ID', 'Target'])
    test = test.drop(columns=['ID'])
    
    print("🔧 Handling missing values...")
    # Logical columns -> "No"
    logical_cols = [
        'has_debit_card', 'has_mobile_money', 'has_loan_account', 'has_insurance',
        'medical_insurance', 'funeral_insurance', 'motor_vehicle_insurance',
        'has_internet_banking', 'has_credit_card'
    ]
    for col in logical_cols:
        if col in train.columns:
            train[col] = train[col].fillna("No")
            test[col] = test[col].fillna("No")
    
    # Categorical -> "Unknown"
    obj_cols = train.select_dtypes(include=['object']).columns
    for col in obj_cols:
        train[col] = train[col].fillna("Unknown")
        test[col] = test[col].fillna("Unknown")
    
    # Numerical -> median
    num_cols = train.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)
    
    print("🔧 Creating interaction features...")
    train = create_interaction_features(train)
    test = create_interaction_features(test)
    
    print("🔧 Creating binned features...")
    train = create_binned_features(train)
    test = create_binned_features(test)
    
    # Target Encoding for high-cardinality categoricals
    if use_target_encoding:
        print("🔧 Applying Target Encoding...")
        # Re-attach target temporarily
        train_with_target = train.copy()
        train_with_target['Target'] = target
        
        target_encode_cols = ['country']  # Can add more if needed
        for col in target_encode_cols:
            if col in train.columns:
                train_encoded, test_encoded = target_encode_feature(
                    train_with_target, test, col, 'Target', smoothing=10
                )
                train[col + '_target_enc'] = train_encoded
                test[col + '_target_enc'] = test_encoded
                # Drop original
                train = train.drop(columns=[col])
                test = test.drop(columns=[col])
    
    print("🔧 Label encoding remaining categoricals...")
    cat_cols = train.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
    
    # Also encode binned categoricals
    for col in ['age_bin', 'income_bin', 'business_age_bin']:
        if col in train.columns:
            le = LabelEncoder()
            combined = pd.concat([train[col].astype(str), test[col].astype(str)], axis=0)
            le.fit(combined)
            train[col] = le.transform(train[col].astype(str))
            test[col] = le.transform(test[col].astype(str))
    
    print(f"✅ Preprocessing complete! Features: {train.shape[1]}")
    
    return train, target, test, train_ids, test_ids
