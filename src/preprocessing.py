"""
Preprocessing Module for Parkinson's Disease Detection
Handles data loading, validation, cleaning, feature engineering, and scaling
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParkinsonDataProcessor:
    """
    Professional data preprocessing pipeline for Parkinson's disease detection.
    Implements comprehensive validation, feature engineering, and scaling strategies.
    """
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize the data processor.
        
        Args:
            scaler_type: Type of scaler ('standard' or 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_names = None
        self.target_name = 'status'
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and validate Parkinson's dataset.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data validation fails
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Validate dataset structure
        self._validate_data(df)
        
        logger.info(f"Data loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate data integrity and structure.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError("Dataset is empty")
        
        if self.target_name not in df.columns:
            raise ValueError(f"Target column '{self.target_name}' not found")
        
        # Check for required vocal features
        required_features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer']
        missing = [f for f in required_features if f not in df.columns]
        if missing:
            logger.warning(f"Missing expected features: {missing}")
        
        # Check data quality
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            logger.warning(f"Dataset contains {null_count} null values")
        
        # Validate target distribution
        target_dist = df[self.target_name].value_counts()
        logger.info(f"Target distribution:\n{target_dist}")
        
        if len(target_dist) != 2:
            raise ValueError("Target must be binary (0: healthy, 1: Parkinson's)")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare data for modeling.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove 'name' column if present (patient identifier)
        if 'name' in df_clean.columns:
            df_clean = df_clean.drop('name', axis=1)
            logger.info("Removed 'name' column")
        
        # Handle missing values
        null_counts = df_clean.isnull().sum()
        if null_counts.any():
            logger.warning("Handling missing values")
            # For healthcare data, use median imputation for robustness
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    logger.info(f"Imputed {col} with median: {median_val:.4f}")
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        # Detect and handle outliers (using IQR method)
        df_clean = self._handle_outliers(df_clean)
        
        return df_clean
    
    def _handle_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect and cap extreme outliers using IQR method.
        
        Args:
            df: DataFrame to process
            threshold: IQR multiplier threshold
            
        Returns:
            DataFrame with capped outliers
        """
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_name]
        
        outlier_counts = {}
        
        for col in numeric_cols:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Count outliers
            outliers = ((df_processed[col] < lower_bound) | 
                       (df_processed[col] > upper_bound)).sum()
            
            if outliers > 0:
                outlier_counts[col] = outliers
                # Cap outliers instead of removing (preserves data)
                df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
        
        if outlier_counts:
            logger.info(f"Capped outliers in {len(outlier_counts)} features")
        
        return df_processed
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features for Parkinson's detection.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df_eng = df.copy()
        
        # Jitter-Shimmer ratio (vocal instability measure)
        if 'MDVP:Jitter(%)' in df_eng.columns and 'MDVP:Shimmer' in df_eng.columns:
            df_eng['jitter_shimmer_ratio'] = (
                df_eng['MDVP:Jitter(%)'] / (df_eng['MDVP:Shimmer'] + 1e-8)
            )
        
        # Harmonicity metrics
        if 'HNR' in df_eng.columns:
            df_eng['hnr_squared'] = df_eng['HNR'] ** 2
        
        # Spread measures interaction
        if 'spread1' in df_eng.columns and 'spread2' in df_eng.columns:
            df_eng['spread_product'] = df_eng['spread1'] * df_eng['spread2']
        
        # RPDE and DFA interaction (nonlinear dynamics)
        if 'RPDE' in df_eng.columns and 'DFA' in df_eng.columns:
            df_eng['rpde_dfa_interaction'] = df_eng['RPDE'] * df_eng['DFA']
        
        # PPE (Pitch Period Entropy) polynomial features
        if 'PPE' in df_eng.columns:
            df_eng['ppe_squared'] = df_eng['PPE'] ** 2
        
        logger.info(f"Engineered {df_eng.shape[1] - df.shape[1]} new features")
        return df_eng
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2, 
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets with stratification.
        
        Args:
            df: Processed DataFrame
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
            stratify: Whether to stratify by target
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X = df.drop(self.target_name, axis=1)
        y = df[self.target_name]
        
        self.feature_names = X.columns.tolist()
        
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_param
        )
        
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        logger.info(f"Train distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using fitted scaler (fit on train, transform both).
        
        Args:
            X_train: Training features
            X_test: Testing features
            
        Returns:
            Tuple of (scaled_X_train, scaled_X_test)
        """
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Features scaled using {self.scaler_type} scaler")
        
        return X_train_scaled, X_test_scaled
    
    def save_scaler(self, filepath: str) -> None:
        """
        Save fitted scaler to disk.
        
        Args:
            filepath: Path to save scaler
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted yet")
        
        joblib.dump(self.scaler, filepath)
        logger.info(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str) -> None:
        """
        Load fitted scaler from disk.
        
        Args:
            filepath: Path to scaler file
        """
        self.scaler = joblib.load(filepath)
        logger.info(f"Scaler loaded from {filepath}")
    
    def get_feature_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive feature statistics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'shape': df.shape,
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'missing_values': df.isnull().sum().to_dict(),
            'target_distribution': df[self.target_name].value_counts().to_dict(),
            'feature_ranges': {
                col: {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }
                for col in df.select_dtypes(include=[np.number]).columns
                if col != self.target_name
            }
        }
        
        return stats


def preprocess_pipeline(
    data_path: str,
    test_size: float = 0.2,
    engineer_features: bool = True,
    scaler_type: str = 'standard',
    save_scaler_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ParkinsonDataProcessor]:
    """
    Complete preprocessing pipeline for Parkinson's disease detection.
    
    Args:
        data_path: Path to CSV data file
        test_size: Test set proportion
        engineer_features: Whether to create engineered features
        scaler_type: Type of scaler to use
        save_scaler_path: Optional path to save scaler
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, processor)
    """
    processor = ParkinsonDataProcessor(scaler_type=scaler_type)
    
    # Load and validate
    df = processor.load_data(data_path)
    
    # Clean data
    df_clean = processor.clean_data(df)
    
    # Engineer features if requested
    if engineer_features:
        df_processed = processor.engineer_features(df_clean)
    else:
        df_processed = df_clean
    
    # Split data
    X_train, X_test, y_train, y_test = processor.split_data(
        df_processed, 
        test_size=test_size
    )
    
    # Scale features
    X_train_scaled, X_test_scaled = processor.scale_features(X_train, X_test)
    
    # Save scaler if path provided
    if save_scaler_path:
        processor.save_scaler(save_scaler_path)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, processor


if __name__ == "__main__":
    # Example usage
    X_train, X_test, y_train, y_test, processor = preprocess_pipeline(
        data_path='data/parkinsons.csv',
        test_size=0.2,
        engineer_features=True,
        scaler_type='standard',
        save_scaler_path='models/scaler.pkl'
    )
    
    print(f"\nPreprocessing complete!")
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    print(f"Features: {len(processor.feature_names)}")