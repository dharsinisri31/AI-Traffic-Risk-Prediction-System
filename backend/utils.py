import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger("ai_traffic_system")

def get_congestion_recommendation(level: str) -> str:
    """
    Returns an actionable recommendation based on traffic congestion level.
    """
    normalized = str(level).strip().capitalize()
    if normalized == "High":
        return "High traffic congestion detected. Consider alternative routes, off-peak travel, or public transit to minimize delays."
    elif normalized in ("Moderate", "Medium"):
        return "Moderate traffic congestion detected. Exercise patience, maintain safe following distance, and allow slight extra travel time."
    elif normalized == "Low":
        return "Traffic flow is normal. Minimal to no delays expected on the route. Continue safe driving practices."
    return "Traffic flow conditions monitored. Maintain standard driving safety."

def get_accident_recommendation(risk_level: str) -> str:
    """
    Returns a safety recommendation based on accident risk estimation.
    """
    normalized = str(risk_level).strip().capitalize()
    if normalized == "High":
        return "Reduce speed, maintain a safe following distance, and consider avoiding high-risk conditions when possible."
    elif normalized in ("Moderate", "Medium"):
        return "Exercise additional caution and maintain a safe following distance."
    elif normalized == "Low":
        return "Traffic and environmental conditions indicate relatively low risk. Continue normal safe driving practices."
    return "Drive carefully and maintain situational awareness."

def prepare_input_dataframe(data: Dict[str, Any], feature_columns: List[str]) -> pd.DataFrame:
    """
    Prepares and aligns input data into a pandas DataFrame matching
    the exact feature columns and ordering required by the trained model.
    """
    # Create single-row dictionary ensuring all expected columns exist
    row_dict = {}
    for col in feature_columns:
        val = data.get(col, 0.0)
        # Convert boolean or None safely
        if val is None:
            val = 0.0
        elif isinstance(val, bool):
            val = 1 if val else 0
        else:
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = 0.0
        row_dict[col] = val

    # Create DataFrame and explicitly re-index to match feature_columns order
    df = pd.DataFrame([row_dict], columns=feature_columns)
    return df
