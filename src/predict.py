"""
Prediction Module for Parkinson's Disease Detection
Production-ready inference pipeline with validation and interpretability
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional, Union, List
import logging
from pathlib import Path
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParkinsonPredictor:
    """
    Production inference system for Parkinson's disease detection.
    Provides predictions, confidence scores, and clinical interpretations.
    """
    
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize the predictor with trained model and scaler.
        
        Args:
            model_path: Path to trained model file
            scaler_path: Path to fitted scaler file
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._load_artifacts()
        
    def _load_artifacts(self) -> None:
        """Load model and scaler from disk."""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"Model loaded from {self.model_path}")
            logger.info(f"Scaler loaded from {self.scaler_path}")
            
            # Load metadata if available
            metadata_path = Path(self.model_path).with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    logger.info(f"Model type: {metadata.get('model_type', 'Unknown')}")
        
        except Exception as e:
            logger.error(f"Error loading artifacts: {str(e)}")
            raise
    
    def validate_input(self, features: Union[np.ndarray, pd.DataFrame, Dict]) -> np.ndarray:
        """
        Validate and prepare input features for prediction.
        
        Args:
            features: Input features (array, DataFrame, or dict)
            
        Returns:
            Validated numpy array
            
        Raises:
            ValueError: If input validation fails
        """
        # Convert to numpy array
        if isinstance(features, dict):
            # Single sample as dictionary
            features = np.array([list(features.values())])
        elif isinstance(features, pd.DataFrame):
            features = features.values
        elif isinstance(features, list):
            features = np.array(features)
        elif not isinstance(features, np.ndarray):
            raise ValueError("Input must be array, DataFrame, dict, or list")
        
        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Validate shape
        if features.shape[1] != self.scaler.n_features_in_:
            raise ValueError(
                f"Expected {self.scaler.n_features_in_} features, "
                f"got {features.shape[1]}"
            )
        
        # Check for NaN or infinite values
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            raise ValueError("Input contains NaN or infinite values")
        
        return features
    
    def predict(
        self, 
        features: Union[np.ndarray, pd.DataFrame, Dict],
        return_proba: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction for Parkinson's disease.
        
        Args:
            features: Input features
            return_proba: Whether to return probability scores
            
        Returns:
            Dictionary containing prediction results
        """
        # Validate input
        features = self.validate_input(features)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get prediction
        prediction = self.model.predict(features_scaled)
        
        # Initialize result dictionary
        result = {
            'prediction': int(prediction[0]),
            'diagnosis': 'Parkinson\'s Disease' if prediction[0] == 1 else 'Healthy',
            'timestamp': datetime.now().isoformat()
        }
        
        # Add probability scores if available and requested
        if return_proba and hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features_scaled)[0]
            result['probability'] = {
                'healthy': float(proba[0]),
                'parkinsons': float(proba[1])
            }
            result['confidence'] = float(max(proba))
        elif return_proba and hasattr(self.model, 'decision_function'):
            decision = self.model.decision_function(features_scaled)[0]
            result['decision_score'] = float(decision)
            result['confidence'] = float(abs(decision))
        
        return result
    
    def predict_batch(
        self,
        features: Union[np.ndarray, pd.DataFrame],
        return_proba: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple samples.
        
        Args:
            features: Input features for multiple samples
            return_proba: Whether to return probability scores
            
        Returns:
            List of prediction dictionaries
        """
        # Validate input
        features = self.validate_input(features)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get predictions
        predictions = self.model.predict(features_scaled)
        
        # Initialize results
        results = []
        
        for i, pred in enumerate(predictions):
            result = {
                'sample_id': i,
                'prediction': int(pred),
                'diagnosis': 'Parkinson\'s Disease' if pred == 1 else 'Healthy'
            }
            
            # Add probabilities if available
            if return_proba and hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features_scaled)[i]
                result['probability'] = {
                    'healthy': float(proba[0]),
                    'parkinsons': float(proba[1])
                }
                result['confidence'] = float(max(proba))
            
            results.append(result)
        
        return results
    
    def get_clinical_interpretation(self, prediction_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate clinical interpretation of prediction results.
        
        Args:
            prediction_result: Result from predict() method
            
        Returns:
            Dictionary with clinical interpretation
        """
        diagnosis = prediction_result['diagnosis']
        confidence = prediction_result.get('confidence', None)
        
        interpretation = {
            'diagnosis': diagnosis,
            'recommendation': ''
        }
        
        if diagnosis == 'Parkinson\'s Disease':
            if confidence and confidence > 0.9:
                interpretation['risk_level'] = 'High'
                interpretation['recommendation'] = (
                    "Strong indication of Parkinson's Disease. "
                    "Immediate consultation with a neurologist is recommended. "
                    "Consider comprehensive neurological examination and motor assessment."
                )
            elif confidence and confidence > 0.7:
                interpretation['risk_level'] = 'Moderate-High'
                interpretation['recommendation'] = (
                    "Moderate to high indication of Parkinson's Disease. "
                    "Medical consultation recommended for detailed evaluation and follow-up."
                )
            else:
                interpretation['risk_level'] = 'Moderate'
                interpretation['recommendation'] = (
                    "Possible indication of Parkinson's Disease. "
                    "Consider medical consultation and monitoring of symptoms."
                )
        else:
            if confidence and confidence > 0.9:
                interpretation['risk_level'] = 'Low'
                interpretation['recommendation'] = (
                    "Low likelihood of Parkinson's Disease based on vocal features. "
                    "Continue routine health monitoring."
                )
            elif confidence and confidence > 0.7:
                interpretation['risk_level'] = 'Low-Moderate'
                interpretation['recommendation'] = (
                    "Results suggest healthy status, but with moderate confidence. "
                    "If symptoms persist, consider follow-up evaluation."
                )
            else:
                interpretation['risk_level'] = 'Uncertain'
                interpretation['recommendation'] = (
                    "Results are uncertain. Additional testing and medical "
                    "consultation recommended for definitive diagnosis."
                )
        
        interpretation['confidence_score'] = f"{confidence*100:.1f}%" if confidence else "N/A"
        
        return interpretation
    
    def explain_prediction(
        self,
        features: Union[np.ndarray, pd.DataFrame, Dict],
        feature_names: Optional[List[str]] = None,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Explain prediction using feature contributions (for linear models).
        
        Args:
            features: Input features
            feature_names: List of feature names
            top_n: Number of top contributing features to show
            
        Returns:
            Dictionary with explanation
        """
        # Validate and scale features
        features = self.validate_input(features)
        features_scaled = self.scaler.transform(features)
        
        explanation = {
            'supported': False,
            'method': None,
            'top_features': []
        }
        
        # For linear models, use coefficients
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_[0]
            contributions = features_scaled[0] * coef
            
            # Get top contributing features
            top_indices = np.argsort(np.abs(contributions))[-top_n:][::-1]
            
            for idx in top_indices:
                feature_name = feature_names[idx] if feature_names else f"Feature_{idx}"
                explanation['top_features'].append({
                    'feature': feature_name,
                    'contribution': float(contributions[idx]),
                    'value': float(features_scaled[0][idx])
                })
            
            explanation['supported'] = True
            explanation['method'] = 'Linear Coefficients'
        
        # For tree-based models, use feature importance
        elif hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[-top_n:][::-1]
            
            for idx in top_indices:
                feature_name = feature_names[idx] if feature_names else f"Feature_{idx}"
                explanation['top_features'].append({
                    'feature': feature_name,
                    'importance': float(importances[idx]),
                    'value': float(features_scaled[0][idx])
                })
            
            explanation['supported'] = True
            explanation['method'] = 'Feature Importance'
        
        return explanation
    
    def save_prediction(self, result: Dict[str, Any], output_path: str) -> None:
        """
        Save prediction result to file.
        
        Args:
            result: Prediction result dictionary
            output_path: Path to save result
        """
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Prediction saved to {output_path}")


def predict_from_file(
    model_path: str,
    scaler_path: str,
    input_file: str,
    output_file: Optional[str] = None,
    generate_interpretation: bool = True
) -> Union[Dict, List[Dict]]:
    """
    Make predictions from input file.
    
    Args:
        model_path: Path to trained model
        scaler_path: Path to fitted scaler
        input_file: Path to input CSV/JSON file
        output_file: Optional path to save results
        generate_interpretation: Whether to generate clinical interpretation
        
    Returns:
        Prediction results
    """
    # Initialize predictor
    predictor = ParkinsonPredictor(model_path, scaler_path)
    
    # Load input data
    file_ext = Path(input_file).suffix.lower()
    
    if file_ext == '.csv':
        data = pd.read_csv(input_file)
        # Remove non-feature columns if present
        if 'name' in data.columns:
            data = data.drop('name', axis=1)
        if 'status' in data.columns:
            data = data.drop('status', axis=1)
        
        # Batch prediction
        results = predictor.predict_batch(data)
        
    elif file_ext == '.json':
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Multiple samples
            results = predictor.predict_batch(np.array(data))
        else:
            # Single sample
            results = predictor.predict(data)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Add clinical interpretation
    if generate_interpretation:
        if isinstance(results, list):
            for result in results:
                interpretation = predictor.get_clinical_interpretation(result)
                result['clinical_interpretation'] = interpretation
        else:
            interpretation = predictor.get_clinical_interpretation(results)
            results['clinical_interpretation'] = interpretation
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    return results


def interactive_prediction(model_path: str, scaler_path: str) -> None:
    """
    Interactive command-line prediction interface.
    
    Args:
        model_path: Path to trained model
        scaler_path: Path to fitted scaler
    """
    predictor = ParkinsonPredictor(model_path, scaler_path)
    
    print("\n" + "="*70)
    print("PARKINSON'S DISEASE DETECTION - INTERACTIVE PREDICTION")
    print("="*70 + "\n")
    
    # Get feature names
    n_features = predictor.scaler.n_features_in_
    print(f"Model expects {n_features} features")
    print("Please enter feature values (comma-separated):\n")
    
    try:
        # Get input from user
        user_input = input("Features: ")
        features = [float(x.strip()) for x in user_input.split(',')]
        
        if len(features) != n_features:
            print(f"\nError: Expected {n_features} features, got {len(features)}")
            return
        
        # Make prediction
        result = predictor.predict(np.array(features))
        
        # Get clinical interpretation
        interpretation = predictor.get_clinical_interpretation(result)
        
        # Display results
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        print(f"\nDiagnosis: {result['diagnosis']}")
        
        if 'probability' in result:
            print(f"\nProbabilities:")
            print(f"  Healthy:     {result['probability']['healthy']*100:.2f}%")
            print(f"  Parkinson's: {result['probability']['parkinsons']*100:.2f}%")
            print(f"\nConfidence: {result['confidence']*100:.2f}%")
        
        print(f"\n{'-'*70}")
        print("CLINICAL INTERPRETATION")
        print(f"{'-'*70}")
        print(f"Risk Level: {interpretation['risk_level']}")
        print(f"Confidence: {interpretation['confidence_score']}")
        print(f"\nRecommendation:")
        print(f"{interpretation['recommendation']}")
        print("="*70 + "\n")
        
        # Ask if user wants to save result
        save = input("Save result to file? (y/n): ")
        if save.lower() == 'y':
            filename = input("Enter filename (default: prediction_result.json): ").strip()
            if not filename:
                filename = "prediction_result.json"
            
            result['clinical_interpretation'] = interpretation
            predictor.save_prediction(result, filename)
            print(f"Result saved to {filename}")
        
    except ValueError as e:
        print(f"\nError: Invalid input - {str(e)}")
    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'interactive':
            # Interactive mode
            interactive_prediction('models/svm_model.pkl', 'models/scaler.pkl')
        elif sys.argv[1] == 'file' and len(sys.argv) > 2:
            # File prediction mode
            input_file = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else 'predictions.json'
            
            results = predict_from_file(
                model_path='models/svm_model.pkl',
                scaler_path='models/scaler.pkl',
                input_file=input_file,
                output_file=output_file
            )
            print(f"\nPredictions completed. Results saved to {output_file}")
        else:
            print("Usage:")
            print("  python predict.py interactive")
            print("  python predict.py file <input_file> [output_file]")
    else:
        # Default: run interactive mode
        interactive_prediction('models/svm_model.pkl', 'models/scaler.pkl')