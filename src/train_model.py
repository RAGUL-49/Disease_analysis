"""
Model Training Module for Parkinson's Disease Detection
Implements multiple ML algorithms with hyperparameter tuning and model selection
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from typing import Dict, Any, Tuple, Optional
import logging
import joblib
from pathlib import Path
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParkinsonModelTrainer:
    """
    Professional model training framework for Parkinson's disease detection.
    Supports multiple algorithms, hyperparameter tuning, and cross-validation.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.training_history = []
        
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Define model configurations and hyperparameter grids.
        
        Returns:
            Dictionary of model configs
        """
        configs = {
            'svm': {
                'model': SVC(random_state=self.random_state, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'kernel': ['rbf', 'linear', 'poly']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear']
                }
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            }
        }
        
        return configs
    
    def train_single_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str,
        use_grid_search: bool = True,
        cv_folds: int = 5,
        scoring: str = 'f1'
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a single model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of model to train
            use_grid_search: Whether to use grid search
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
            
        Returns:
            Tuple of (trained_model, training_info)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        configs = self.get_model_configs()
        
        if model_name not in configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = configs[model_name]
        model = config['model']
        
        start_time = time.time()
        
        if use_grid_search:
            # Hyperparameter tuning with GridSearchCV
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=config['params'],
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_score = grid_search.best_score_
            
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best CV {scoring} score: {best_cv_score:.4f}")
        else:
            # Train with default parameters
            model.fit(X_train, y_train)
            best_model = model
            best_params = model.get_params()
            best_cv_score = None
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            best_model, X_train, y_train, 
            cv=cv_folds, scoring=scoring
        )
        
        training_time = time.time() - start_time
        
        training_info = {
            'model_name': model_name,
            'best_params': best_params,
            'cv_mean_score': float(np.mean(cv_scores)),
            'cv_std_score': float(np.std(cv_scores)),
            'best_cv_score': float(best_cv_score) if best_cv_score else None,
            'training_time': training_time,
            'cv_folds': cv_folds,
            'scoring_metric': scoring
        }
        
        logger.info(f"CV {scoring} scores: {cv_scores}")
        logger.info(f"Mean CV score: {training_info['cv_mean_score']:.4f} (+/- {training_info['cv_std_score']:.4f})")
        logger.info(f"Training time: {training_time:.2f} seconds")
        
        self.models[model_name] = best_model
        self.training_history.append(training_info)
        
        return best_model, training_info
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        models_to_train: Optional[list] = None,
        use_grid_search: bool = True,
        cv_folds: int = 5,
        scoring: str = 'f1'
    ) -> Dict[str, Any]:
        """
        Train multiple models and compare performance.
        
        Args:
            X_train: Training features
            y_train: Training labels
            models_to_train: List of model names (None = all models)
            use_grid_search: Whether to use grid search
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
            
        Returns:
            Dictionary of training results
        """
        configs = self.get_model_configs()
        
        if models_to_train is None:
            models_to_train = list(configs.keys())
        
        results = {}
        
        for model_name in models_to_train:
            try:
                model, info = self.train_single_model(
                    X_train, y_train,
                    model_name=model_name,
                    use_grid_search=use_grid_search,
                    cv_folds=cv_folds,
                    scoring=scoring
                )
                results[model_name] = info
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        # Select best model
        self._select_best_model(scoring)
        
        return results
    
    def _select_best_model(self, metric: str = 'f1') -> None:
        """
        Select the best performing model based on CV scores.
        
        Args:
            metric: Metric to use for selection
        """
        if not self.training_history:
            logger.warning("No models trained yet")
            return
        
        # Sort by CV mean score
        sorted_models = sorted(
            self.training_history,
            key=lambda x: x['cv_mean_score'],
            reverse=True
        )
        
        best = sorted_models[0]
        self.best_model_name = best['model_name']
        self.best_model = self.models[self.best_model_name]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"BEST MODEL SELECTED: {self.best_model_name.upper()}")
        logger.info(f"CV Score: {best['cv_mean_score']:.4f} (+/- {best['cv_std_score']:.4f})")
        logger.info(f"{'='*60}\n")
    
    def evaluate_on_test(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a trained model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            model_name: Name of model to evaluate (None = best model)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No model trained yet")
            model = self.best_model
            model_name = self.best_model_name
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not trained")
            model = self.models[model_name]
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(y_test, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        logger.info(f"\nTest Set Evaluation - {model_name.upper()}")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"Specificity: {metrics['specificity']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        
        return metrics
    
    @staticmethod
    def _calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate specificity (true negative rate).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Specificity score
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return float(specificity)
    
    def save_model(self, filepath: str, model_name: Optional[str] = None) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model
            model_name: Name of model to save (None = best model)
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No model trained yet")
            model = self.best_model
            model_name = self.best_model_name
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not trained")
            model = self.models[model_name]
        
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Save metadata
        metadata_path = Path(filepath).with_suffix('.json')
        metadata = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'training_history': self.training_history
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to {metadata_path}")
    
    def load_model(self, filepath: str) -> Any:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model
        """
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def get_training_summary(self) -> pd.DataFrame:
        """
        Get summary of all trained models.
        
        Returns:
            DataFrame with training results
        """
        if not self.training_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.training_history)
        df = df.sort_values('cv_mean_score', ascending=False)
        return df


def train_parkinson_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = 'svm',
    use_grid_search: bool = True,
    cv_folds: int = 5,
    save_path: Optional[str] = None
) -> Tuple[Any, Dict[str, float]]:
    """
    Complete training pipeline for Parkinson's disease detection.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_type: Type of model to train
        use_grid_search: Whether to use hyperparameter tuning
        cv_folds: Number of CV folds
        save_path: Optional path to save model
        
    Returns:
        Tuple of (trained_model, test_metrics)
    """
    trainer = ParkinsonModelTrainer()
    
    # Train model
    model, train_info = trainer.train_single_model(
        X_train, y_train,
        model_name=model_type,
        use_grid_search=use_grid_search,
        cv_folds=cv_folds
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate_on_test(X_test, y_test, model_name=model_type)
    
    # Save if path provided
    if save_path:
        trainer.save_model(save_path, model_name=model_type)
    
    return model, test_metrics


if __name__ == "__main__":
    # Example usage
    from preprocessing import preprocess_pipeline
    
    # Preprocess data
    X_train, X_test, y_train, y_test, _ = preprocess_pipeline(
        data_path='data/parkinsons.csv',
        test_size=0.2
    )
    
    # Train single model
    model, metrics = train_parkinson_model(
        X_train, y_train, X_test, y_test,
        model_type='svm',
        use_grid_search=True,
        save_path='models/svm_model.pkl'
    )
    
    # Or train all models and compare
    trainer = ParkinsonModelTrainer()
    results = trainer.train_all_models(
        X_train, y_train,
        use_grid_search=True,
        cv_folds=5
    )
    
    # Get summary
    summary = trainer.get_training_summary()
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(summary[['model_name', 'cv_mean_score', 'cv_std_score', 'training_time']])