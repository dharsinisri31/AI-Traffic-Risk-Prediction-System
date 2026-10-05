"""
AI Traffic Risk Prediction System - Preprocessing and Utility Functions

This module provides data transformation, one-hot encoding alignment,
and domain-specific recommendations for congestion and accident risk prediction.
"""

import logging
from typing import Dict, Any, List
import pandas as pd

# Configure logger for utility operations
logger = logging.getLogger("ai_traffic_system.utils")


def prepare_input_dataframe(input_dict: Dict[str, Any], expected_columns: List[str]) -> pd.DataFrame:
    """
    Transforms and aligns raw input dictionary into a pandas DataFrame that exactly
    matches the feature columns and ordering required by the trained ML models.

    Parameters:
        input_dict (dict): Incoming key-value pairs from the request.
        expected_columns (list): The list of 73 feature names loaded from the PKL file.

    Returns:
        pd.DataFrame: A 1-row DataFrame containing all expected columns in exact order.
    """
    # Start with a dictionary initialized with 0.0 for all expected columns
    row_data: Dict[str, Any] = {col: 0.0 for col in expected_columns}

    # Extract high-level categorical strings if provided
    holiday_val = input_dict.get("holiday")
    weather_main_val = input_dict.get("weather_main")
    weather_desc_val = input_dict.get("weather_description")

    # If categorical strings are present, set corresponding one-hot encoded flags to 1
    if holiday_val is not None:
        holiday_col = f"holiday_{holiday_val}"
        if holiday_col in row_data:
            row_data[holiday_col] = 1.0
        elif holiday_val.lower() == "none" and "holiday_None" in row_data:
            row_data["holiday_None"] = 1.0

    if weather_main_val is not None:
        weather_main_col = f"weather_main_{weather_main_val}"
        if weather_main_col in row_data:
            row_data[weather_main_col] = 1.0

    if weather_desc_val is not None:
        weather_desc_col = f"weather_description_{weather_desc_val}"
        if weather_desc_col in row_data:
            row_data[weather_desc_col] = 1.0

    # Overlay all numeric and direct feature keys passed in the input
    for key, val in input_dict.items():
        if val is None:
            continue

        # If key directly matches an expected feature column
        if key in row_data:
            try:
                row_data[key] = float(val)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert key {key} with value {val} to float. Defaulting to 0.0.")
                row_data[key] = 0.0
        
        # Handle field name conversions where spaces might be underscores in request body
        normalized_key = key.replace("_", " ")
        if normalized_key in row_data:
            try:
                row_data[normalized_key] = float(val)
            except (ValueError, TypeError):
                row_data[normalized_key] = 0.0

    # Ensure holiday_None defaults to 1 if no holiday flag was activated
    holiday_active = any(row_data.get(f"holiday_{h}", 0.0) == 1.0 for h in [
        "Columbus Day", "Independence Day", "Labor Day", "Martin Luther King Jr Day",
        "Memorial Day", "New Years Day", "State Fair", "Thanksgiving Day",
        "Veterans Day", "Washingtons Birthday"
    ])
    if not holiday_active and "holiday_None" in row_data:
        row_data["holiday_None"] = 1.0

    # Create DataFrame with exact column ordering
    df = pd.DataFrame([row_data], columns=expected_columns)
    return df


def get_congestion_recommendation(congestion_level: str) -> str:
    """
    Returns practical traffic management recommendations based on predicted congestion class.
    """
    level = str(congestion_level).strip().capitalize()
    if level == "High":
        return "Heavy congestion expected. Recommend rerouting traffic, activating dynamic signal timing, and advising motorists to seek alternate routes."
    elif level == "Medium":
        return "Moderate congestion expected. Monitor key corridors and maintain standard peak flow management."
    elif level == "Low":
        return "Free-flowing traffic anticipated. No special traffic intervention required."
    return "Standard traffic monitoring advised."


def get_accident_recommendation(risk_level: str) -> str:
    """
    Returns road safety guidance based on estimated accident risk class.
    """
    level = str(risk_level).strip().capitalize()
    if level == "High":
        return "High estimated accident risk. Recommend dispatching safety patrols, warning variable message signs (VMS), and lowering variable speed limits."
    elif level == "Moderate":
        return "Moderate accident risk. Advise drivers to exercise caution, maintain safe following distance, and watch for changing weather conditions."
    elif level == "Low":
        return "Low estimated accident risk. Standard road safety guidelines apply."
    return "Exercise regular safe driving practices."
