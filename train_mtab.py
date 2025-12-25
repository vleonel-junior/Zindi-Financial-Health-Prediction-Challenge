import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
import warnings
import os

warnings.filterwarnings('ignore')

SEED = 42
TARGET_COL = 'Target'
ID_COL = 'ID'
TIME_LIMIT = None 
SAVE_PATH = 'ag_models_mitra'

class DataProcessor:
    """
    Handles data loading, preprocessing, and feature engineering for the Zindi Financial Health challenge.
    """
    
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the Train, Test, and SampleSubmission datasets.
        
        Attempts to load from the local directory first. If failed, recursively searches 
        the '/kaggle/input' directory for valid files.

        Returns:
            tuple: (train_df, test_df, submission_df)
        
        Raises:
            FileNotFoundError: If datasets cannot be located in local or Kaggle paths.
        """
        print("⏳ Loading Data...")
        try:
            return pd.read_csv('Train.csv'), pd.read_csv('Test.csv'), pd.read_csv('SampleSubmission.csv')
        except FileNotFoundError:
            try:
                base_dir = '/kaggle/input'
                for root, dirs, files in os.walk(base_dir):
                    if 'Train.csv' in files:
                        p = lambda x: os.path.join(root, x)
                        print(f"✅ Found data in {root}")
                        return pd.read_csv(p('Train.csv')), pd.read_csv(p('Test.csv')), pd.read_csv(p('SampleSubmission.csv'))
                raise FileNotFoundError
            except:
                raise Exception("❌ Files not found. Please ensure the dataset is uploaded.")

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cleaning, logical imputation, and feature engineering.

        Args:
            df (pd.DataFrame): Input dataframe (Train or Test).

        Returns:
            pd.DataFrame: Processed dataframe ready for modeling.
        """
        df = df.copy()
        
        # Logical Imputation: Treat missing binary financial indicators as 'No'
        logical_cols = [
            'has_debit_card', 'has_mobile_money', 'has_loan_account', 'has_insurance',
            'medical_insurance', 'funeral_insurance', 'motor_vehicle_insurance',
            'has_internet_banking', 'has_credit_card'
        ]
        for col in logical_cols:
            if col in df.columns:
                df[col] = df[col].fillna("No")

        # Categorical Handling: Fill remaining objects with "Unknown"
        obj_cols = df.select_dtypes(include=['object']).columns
        cols_to_fill = [c for c in obj_cols if c not in [ID_COL, TARGET_COL]]
        df[cols_to_fill] = df[cols_to_fill].fillna("Unknown")
        
        # Feature Engineering: Financial Ratios
        if 'personal_income' in df.columns and 'owner_age' in df.columns:
            df['income_per_age'] = df['personal_income'] / (df['owner_age'] + 1)
            
        return df

def main():
    """
    Main execution pipeline.
    
    1. Loads data.
    2. preprocesses training and testing sets.
    3. Trains an AutoGluon predictor using the MITRA foundation model configuration.
    4. Generates a submission file.
    """
    processor = DataProcessor()
    
    train_raw, test_raw, ss = processor.load_data()
    train_clean = processor.preprocess(train_raw)
    test_clean = processor.preprocess(test_raw)
    
    train_data = TabularDataset(train_clean.drop(columns=[ID_COL]))
    test_data = TabularDataset(test_clean.drop(columns=[ID_COL]))
    
    print(f"📊 Shapes: Train {train_data.shape}, Test {test_data.shape}")
    
    predictor = TabularPredictor(
        label=TARGET_COL, 
        eval_metric='f1_macro',
        path=SAVE_PATH
    )
    
    # Configure AutoGluon to use Mitra with Fine-Tuning (Optimized for Kaggle GPU)
    predictor.fit(
        train_data,
        hyperparameters={
            'MITRA': {
                'fine_tune': True, 
                'fine_tune_steps': 5,  # Reduced from 10 to save memory
                'max_samples_support': 512  # Explicitly set low to avoid OOM
            }
        },
        time_limit=TIME_LIMIT,
        presets='medium_quality',  # Changed from best_quality to reduce bagging
        num_bag_folds=3,  # Reduced from default 8 to save memory
        num_bag_sets=1,
        num_stack_levels=0,  # Disable stacking to save memory
        ag_args_fit={
            'ag.max_memory_usage_ratio': 0.7  # Use only 70% of available memory as safety buffer
        }
    )
    
    print("✅ Training Completed!")
    
    preds = predictor.predict(test_data)
    
    submission = pd.DataFrame({ID_COL: test_raw[ID_COL], TARGET_COL: preds})
    submission.to_csv('submission_mitra.csv', index=False)
    print("✅ Saved to submission_mitra.csv")

if __name__ == "__main__":
    main()
