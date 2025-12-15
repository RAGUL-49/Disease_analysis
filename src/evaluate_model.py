"""
Model Evaluation Module for Parkinson's Disease Detection
Comprehensive evaluation with visualizations and clinical metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve, validation_curve
from typing import Dict, Any, Optional, Tuple
import logging
import joblib
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


class ParkinsonModelEvaluator:
    """
    Comprehensive evaluation framework for Parkinson's disease detection models.
    Includes clinical metrics, visualizations, and interpretability analysis.
    """
    
    def __init__(self, model, X_test: np.ndarray, y_test: np.ndarray):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.y_pred_proba = None
        self.metrics = {}
        
    def predict(self) -> None:
        """Generate predictions on test set."""
        self.y_pred = self.model.predict(self.X_test)
        
        if hasattr(self.model, 'predict_proba'):
            self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        elif hasattr(self.model, 'decision_function'):
            self.y_pred_proba = self.model.decision_function(self.X_test)
        else:
            self.y_pred_proba = None
        
        logger.info("Predictions generated")
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Returns:
            Dictionary of metrics
        """
        if self.y_pred is None:
            self.predict()
        
        # Standard classification metrics
        self.metrics['accuracy'] = accuracy_score(self.y_test, self.y_pred)
        self.metrics['precision'] = precision_score(self.y_test, self.y_pred, zero_division=0)
        self.metrics['recall'] = recall_score(self.y_test, self.y_pred, zero_division=0)
        self.metrics['f1_score'] = f1_score(self.y_test, self.y_pred, zero_division=0)
        
        # Confusion matrix components
        cm = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Clinical metrics
        self.metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        self.metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        self.metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        self.metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # False positive and negative rates
        self.metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        self.metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Probability-based metrics
        if self.y_pred_proba is not None:
            self.metrics['roc_auc'] = roc_auc_score(self.y_test, self.y_pred_proba)
            self.metrics['average_precision'] = average_precision_score(self.y_test, self.y_pred_proba)
        
        # Diagnostic metrics
        self.metrics['diagnostic_odds_ratio'] = (
            (tp * tn) / (fp * fn) if (fp * fn) > 0 else float('inf')
        )
        
        self.metrics['confusion_matrix'] = cm.tolist()
        
        return self.metrics
    
    def print_classification_report(self) -> None:
        """Print detailed classification report."""
        if self.y_pred is None:
            self.predict()
        
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT - PARKINSON'S DISEASE DETECTION")
        print("="*70)
        
        report = classification_report(
            self.y_test, self.y_pred,
            target_names=['Healthy', 'Parkinson\'s'],
            digits=4
        )
        print(report)
        
        print("\nCLINICAL METRICS")
        print("-"*70)
        if self.metrics:
            print(f"Sensitivity (Recall):          {self.metrics['sensitivity']:.4f}")
            print(f"Specificity:                   {self.metrics['specificity']:.4f}")
            print(f"Positive Predictive Value:     {self.metrics['positive_predictive_value']:.4f}")
            print(f"Negative Predictive Value:     {self.metrics['negative_predictive_value']:.4f}")
            print(f"False Positive Rate:           {self.metrics['false_positive_rate']:.4f}")
            print(f"False Negative Rate:           {self.metrics['false_negative_rate']:.4f}")
            if 'roc_auc' in self.metrics:
                print(f"ROC-AUC Score:                 {self.metrics['roc_auc']:.4f}")
        
        print("="*70 + "\n")
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix heatmap.
        
        Args:
            save_path: Optional path to save figure
        """
        if self.y_pred is None:
            self.predict()
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Parkinson\'s'],
            yticklabels=['Healthy', 'Parkinson\'s'],
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Parkinson\'s Disease Detection', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plot ROC curve.
        
        Args:
            save_path: Optional path to save figure
        """
        if self.y_pred_proba is None:
            logger.warning("Model does not support probability predictions")
            return
        
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Parkinson\'s Disease Detection', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            save_path: Optional path to save figure
        """
        if self.y_pred_proba is None:
            logger.warning("Model does not support probability predictions")
            return
        
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)
        avg_precision = average_precision_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve - Parkinson\'s Disease Detection', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curve saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Plot distribution of prediction probabilities.
        
        Args:
            save_path: Optional path to save figure
        """
        if self.y_pred_proba is None:
            logger.warning("Model does not support probability predictions")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Separate probabilities by true class
        healthy_probs = self.y_pred_proba[self.y_test == 0]
        parkinsons_probs = self.y_pred_proba[self.y_test == 1]
        
        plt.hist(healthy_probs, bins=30, alpha=0.6, label='Healthy', color='green', edgecolor='black')
        plt.hist(parkinsons_probs, bins=30, alpha=0.6, label='Parkinson\'s', color='red', edgecolor='black')
        
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Prediction Probabilities', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction distribution saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(
        self, 
        feature_names: Optional[list] = None,
        top_n: int = 15,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot feature importance (if model supports it).
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to display
            save_path: Optional path to save figure
        """
        importance = None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            logger.warning("Model does not support feature importance")
            return
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importance))]
        
        # Sort features by importance
        indices = np.argsort(importance)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importance = importance[indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importance, color='steelblue', edgecolor='black')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, output_dir: str = 'reports') -> None:
        """
        Generate complete evaluation report with all visualizations.
        
        Args:
            output_dir: Directory to save report files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating evaluation report in {output_dir}")
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Print report
        self.print_classification_report()
        
        # Generate visualizations
        self.plot_confusion_matrix(save_path=output_path / 'confusion_matrix.png')
        self.plot_roc_curve(save_path=output_path / 'roc_curve.png')
        self.plot_precision_recall_curve(save_path=output_path / 'precision_recall_curve.png')
        self.plot_prediction_distribution(save_path=output_path / 'prediction_distribution.png')
        
        # Save metrics to JSON
        metrics_file = output_path / 'metrics.json'
        with open(metrics_file, 'w') as f:
            # Convert numpy types to Python native types
            serializable_metrics = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in self.metrics.items()
            }
            json.dump(serializable_metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_file}")
        logger.info("Evaluation report generation complete")
    
    def compare_models(
        self,
        models_dict: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models performance.
        
        Args:
            models_dict: Dictionary of model_name: model
            X_test: Test features
            y_test: Test labels
            save_path: Optional path to save comparison plot
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for name, model in models_dict.items():
            evaluator = ParkinsonModelEvaluator(model, X_test, y_test)
            metrics = evaluator.calculate_metrics()
            
            results.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'Specificity': metrics['specificity'],
                'ROC-AUC': metrics.get('roc_auc', np.nan)
            })
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        # Plot comparison
        if save_path:
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(comparison_df))
            width = 0.15
            
            for i, metric in enumerate(metrics_to_plot):
                ax.bar(x + i*width, comparison_df[metric], width, label=metric)
            
            ax.set_xlabel('Models', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x + width * 2)
            ax.set_xticklabels(comparison_df['Model'])
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
            plt.show()
        
        return comparison_df


def evaluate_saved_model(
    model_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[list] = None,
    output_dir: str = 'reports'
) -> Dict[str, float]:
    """
    Load and evaluate a saved model.
    
    Args:
        model_path: Path to saved model
        X_test: Test features
        y_test: Test labels
        feature_names: Optional list of feature names
        output_dir: Directory for evaluation reports
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load model
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    
    # Create evaluator
    evaluator = ParkinsonModelEvaluator(model, X_test, y_test)
    
    # Generate complete report
    evaluator.generate_evaluation_report(output_dir=output_dir)
    
    # Plot feature importance if possible
    if feature_names:
        evaluator.plot_feature_importance(
            feature_names=feature_names,
            save_path=Path(output_dir) / 'feature_importance.png'
        )
    
    return evaluator.metrics


if __name__ == "__main__":
    # Example usage
    from preprocessing import preprocess_pipeline
    from train_model import ParkinsonModelTrainer
    
    # Preprocess data
    X_train, X_test, y_train, y_test, processor = preprocess_pipeline(
        data_path='data/parkinsons.csv',
        test_size=0.2
    )
    
    # Train model
    trainer = ParkinsonModelTrainer()
    model, _ = trainer.train_single_model(
        X_train, y_train,
        model_name='svm',
        use_grid_search=True
    )
    
    # Evaluate model
    evaluator = ParkinsonModelEvaluator(model, X_test, y_test)
    evaluator.generate_evaluation_report(output_dir='reports')
    
    # Plot feature importance
    evaluator.plot_feature_importance(
        feature_names=processor.feature_names,
        top_n=15
    )