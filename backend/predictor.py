import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import joblib
import numpy as np
import pandas as pd

# Handle scikit-learn Cython unpickle compatibility if needed
try:
    import sklearn._loss._loss
    if '_loss' not in sys.modules:
        sys.modules['_loss'] = sklearn._loss._loss
except Exception:
    pass

from backend.utils import (
    get_congestion_recommendation,
    get_accident_recommendation,
    prepare_input_dataframe,
)

logger = logging.getLogger("ai_traffic_system")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

class ModelLoadError(Exception):
    """Raised when models or feature columns fail to load."""
    pass

class PredictionExecutionError(Exception):
    """Raised when a prediction fails during inference."""
    pass

class TrafficModelPredictor:
    """
    Singleton predictor managing loaded machine learning models and feature schemas.
    Loads models once at initialization to prevent redundant disk I/O.
    """
    _instance: Optional["TrafficModelPredictor"] = None

    def __new__(cls) -> "TrafficModelPredictor":
        if cls._instance is None:
            cls._instance = super(TrafficModelPredictor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.congestion_model = None
        self.accident_model = None
        self.congestion_features: List[str] = []
        self.accident_features: List[str] = []
        self._load_all_artifacts()
        self._initialized = True

    def _load_all_artifacts(self) -> None:
        """
        Loads the four pre-trained model and feature-column PKL files.
        """
        congestion_model_path = MODELS_DIR / "congestion_model.pkl"
        accident_model_path = MODELS_DIR / "accident_risk_model.pkl"
        congestion_cols_path = MODELS_DIR / "congestion_feature_columns.pkl"
        accident_cols_path = MODELS_DIR / "accident_feature_columns.pkl"

        for p in [congestion_model_path, accident_model_path, congestion_cols_path, accident_cols_path]:
            if not p.exists():
                raise ModelLoadError(f"Required model artifact not found: {p.name}")

        try:
            logger.info("Loading feature column PKL files...")
            self.congestion_features = joblib.load(congestion_cols_path)
            self.accident_features = joblib.load(accident_cols_path)

            if not isinstance(self.congestion_features, list) or not isinstance(self.accident_features, list):
                raise ModelLoadError("Feature column artifacts must be lists of column names.")

            logger.info(f"Loaded {len(self.congestion_features)} congestion features and {len(self.accident_features)} accident features.")

            logger.info("Loading trained ML models...")
            self.congestion_model = joblib.load(congestion_model_path)
            self.accident_model = joblib.load(accident_model_path)
            logger.info("All machine learning models and feature schemas loaded successfully.")
        except Exception as e:
            logger.error(f"Error during model loading: {e}", exc_info=True)
            raise ModelLoadError(f"Failed to load machine learning models from backend/models/: {str(e)}")

    def predict_congestion(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts traffic congestion level from input feature dictionary.
        Returns:
            {
                "congestion_level": str,
                "confidence": float | None,
                "recommendation": str
            }
        """
        if self.congestion_model is None or not self.congestion_features:
            raise PredictionExecutionError("Congestion model is not loaded.")

        try:
            df = prepare_input_dataframe(input_data, self.congestion_features)
            raw_prediction = self.congestion_model.predict(df)[0]
            congestion_level = str(raw_prediction)

            confidence = None
            if hasattr(self.congestion_model, "predict_proba"):
                proba = self.congestion_model.predict_proba(df)[0]
                confidence = float(np.max(proba))
                confidence = round(confidence, 4)

            recommendation = get_congestion_recommendation(congestion_level)

            return {
                "congestion_level": congestion_level,
                "confidence": confidence,
                "recommendation": recommendation,
            }
        except Exception as e:
            logger.error(f"Congestion prediction error: {e}", exc_info=True)
            raise PredictionExecutionError(f"Failed to compute congestion prediction: {str(e)}")

    def predict_accident_risk(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts accident risk level from input feature dictionary.
        Returns:
            {
                "risk_level": str,
                "risk_percentage": float | None,
                "recommendation": str
            }
        """
        if self.accident_model is None or not self.accident_features:
            raise PredictionExecutionError("Accident risk model is not loaded.")

        try:
            df = prepare_input_dataframe(input_data, self.accident_features)
            raw_prediction = self.accident_model.predict(df)[0]
            risk_level = str(raw_prediction)

            risk_percentage = None
            if hasattr(self.accident_model, "predict_proba"):
                proba = self.accident_model.predict_proba(df)[0]
                # Probability of the predicted class or max probability represented as a percentage
                risk_percentage = round(float(np.max(proba)) * 100.0, 2)

            recommendation = get_accident_recommendation(risk_level)

            return {
                "risk_level": risk_level,
                "risk_percentage": risk_percentage,
                "recommendation": recommendation,
            }
        except Exception as e:
            logger.error(f"Accident risk prediction error: {e}", exc_info=True)
            raise PredictionExecutionError(f"Failed to compute accident risk estimation: {str(e)}")

# Global singleton instance helper
def get_predictor() -> TrafficModelPredictor:
    return TrafficModelPredictor()
