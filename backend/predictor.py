"""
AI Traffic Risk Prediction System - Model Predictor Engine

This module is responsible for:
1. Locating and loading the saved scikit-learn models and feature column lists via joblib.
2. Validating the existence of all four required artifacts.
3. Preprocessing user payloads to match the training feature schema.
4. Generating predictions and probability confidence scores.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import joblib
import numpy as np
import pandas as pd

# Handle scikit-learn Cython unpickle compatibility across minor versions if needed
try:
    import sklearn._loss._loss
    if '_loss' not in sys.modules:
        sys.modules['_loss'] = sklearn._loss._loss
except Exception:
    pass

from backend.utils import (
    prepare_input_dataframe,
    get_congestion_recommendation,
    get_accident_recommendation,
)

# Set up dedicated logger
logger = logging.getLogger("ai_traffic_system.predictor")

# Resolve model directory path
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


class ModelArtifactMissingError(Exception):
    """Raised when a required .pkl model or feature column file is missing on disk."""
    pass


class ModelLoadError(Exception):
    """Raised when an error occurs during unpickling or loading with joblib."""
    pass


class PredictionExecutionError(Exception):
    """Raised when feature preparation or model inference fails."""
    pass


class TrafficModelPredictor:
    """
    Singleton class that loads and holds the trained Machine Learning models in memory.
    This ensures models are loaded only once during application startup, saving CPU and RAM.
    """
    _instance: Optional["TrafficModelPredictor"] = None

    def __new__(cls) -> "TrafficModelPredictor":
        if cls._instance is None:
            cls._instance = super(TrafficModelPredictor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.congestion_model = None
        self.accident_model = None
        self.congestion_features: List[str] = []
        self.accident_features: List[str] = []

        # Load artifacts
        self.load_all_artifacts()
        self._initialized = True

    def load_all_artifacts(self) -> None:
        """
        Loads the 4 pre-trained pickle artifacts from backend/models/:
        1. congestion_model.pkl
        2. congestion_feature_columns.pkl
        3. accident_risk_model.pkl
        4. accident_feature_columns.pkl
        """
        congestion_model_path = MODELS_DIR / "congestion_model.pkl"
        congestion_cols_path = MODELS_DIR / "congestion_feature_columns.pkl"
        accident_model_path = MODELS_DIR / "accident_risk_model.pkl"
        accident_cols_path = MODELS_DIR / "accident_feature_columns.pkl"

        # Fallback for accident columns if named with ' (1)'
        if not accident_cols_path.exists():
            alt_path = MODELS_DIR / "accident_feature_columns (1).pkl"
            if alt_path.exists():
                accident_cols_path = alt_path

        # Verify existence of each file
        for path, name in [
            (congestion_model_path, "congestion_model.pkl"),
            (congestion_cols_path, "congestion_feature_columns.pkl"),
            (accident_model_path, "accident_risk_model.pkl"),
            (accident_cols_path, "accident_feature_columns.pkl"),
        ]:
            if not path.exists():
                err_msg = f"Missing required model artifact: '{name}' in '{MODELS_DIR}'"
                logger.error(err_msg)
                raise ModelArtifactMissingError(err_msg)

        try:
            logger.info("Loading feature column PKL files using joblib...")
            self.congestion_features = joblib.load(congestion_cols_path)
            self.accident_features = joblib.load(accident_cols_path)

            if not isinstance(self.congestion_features, list) or not isinstance(self.accident_features, list):
                raise ModelLoadError("Feature column artifacts must be Python lists of column strings.")

            logger.info(
                f"Feature columns loaded successfully: "
                f"{len(self.congestion_features)} congestion features, {len(self.accident_features)} accident features."
            )

            logger.info("Loading trained ML models using joblib...")
            self.congestion_model = joblib.load(congestion_model_path)
            self.accident_model = joblib.load(accident_model_path)

            logger.info("All Machine Learning models and feature schemas loaded successfully.")

        except Exception as e:
            if isinstance(e, ModelArtifactMissingError):
                raise
            err_msg = f"Failed to load ML models from '{MODELS_DIR}': {str(e)}"
            logger.error(err_msg, exc_info=True)
            raise ModelLoadError(err_msg) from e

    def predict_congestion(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes traffic congestion level prediction.

        Parameters:
            input_data (dict): Dictionary of traffic & weather features.

        Returns:
            dict containing 'congestion_level', 'confidence', and 'recommendation'.
        """
        if self.congestion_model is None or not self.congestion_features:
            raise PredictionExecutionError("Congestion model or feature columns are not loaded.")

        try:
            # Step 1: Preprocess input into exact DataFrame matching trained feature structure
            df = prepare_input_dataframe(input_data, self.congestion_features)

            # Step 2: Make class prediction
            raw_pred = self.congestion_model.predict(df)[0]
            congestion_level = str(raw_pred)

            # Step 3: Compute confidence probability if supported by model
            confidence = None
            if hasattr(self.congestion_model, "predict_proba"):
                probas = self.congestion_model.predict_proba(df)[0]
                confidence = round(float(np.max(probas)), 4)

            # Step 4: Get actionable operational recommendation
            recommendation = get_congestion_recommendation(congestion_level)

            return {
                "congestion_level": congestion_level,
                "confidence": confidence,
                "recommendation": recommendation,
            }

        except Exception as e:
            logger.error(f"Error computing congestion prediction: {e}", exc_info=True)
            raise PredictionExecutionError(f"Congestion prediction failed: {str(e)}") from e

    def predict_accident_risk(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes estimated accident risk prediction.

        Parameters:
            input_data (dict): Dictionary of traffic & weather features.

        Returns:
            dict containing 'risk_level', 'risk_percentage', 'recommendation', and 'disclaimer'.
        """
        if self.accident_model is None or not self.accident_features:
            raise PredictionExecutionError("Accident risk model or feature columns are not loaded.")

        try:
            # Step 1: Preprocess input into exact DataFrame matching trained feature structure
            df = prepare_input_dataframe(input_data, self.accident_features)

            # Step 2: Make class prediction
            raw_pred = self.accident_model.predict(df)[0]
            risk_level = str(raw_pred)

            # Step 3: Compute probability percentage if supported by model
            risk_percentage = None
            if hasattr(self.accident_model, "predict_proba"):
                probas = self.accident_model.predict_proba(df)[0]
                risk_percentage = round(float(np.max(probas)) * 100.0, 2)

            # Step 4: Get safety guidance recommendation
            recommendation = get_accident_recommendation(risk_level)

            return {
                "risk_level": risk_level,
                "risk_percentage": risk_percentage,
                "recommendation": recommendation,
                "disclaimer": (
                    "This prediction is an AI-based ESTIMATED ACCIDENT RISK for decision-support "
                    "and does not guarantee accident occurrence."
                ),
            }

        except Exception as e:
            logger.error(f"Error computing accident risk prediction: {e}", exc_info=True)
            raise PredictionExecutionError(f"Accident risk estimation failed: {str(e)}") from e


# Global accessor for the singleton predictor instance
def get_predictor() -> TrafficModelPredictor:
    """Returns the singleton TrafficModelPredictor instance."""
    return TrafficModelPredictor()
