"""
AI Traffic Risk Prediction System - Pydantic Request & Response Schemas

This module defines data contracts and validation models for incoming API payloads
and outgoing prediction responses.
"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class HealthResponse(BaseModel):
    """Schema for /health endpoint response."""
    status: str = Field(default="healthy", description="Overall health status of the API service")
    service: str = Field(default="AI Traffic Risk Prediction System API", description="Service name")
    models_loaded: bool = Field(default=True, description="Whether ML models are loaded and ready")
    congestion_features_count: int = Field(default=73, description="Number of congestion feature columns")
    accident_features_count: int = Field(default=73, description="Number of accident feature columns")


class CongestionPredictionResponse(BaseModel):
    """Schema for /predict/congestion response."""
    congestion_level: str = Field(..., description="Predicted congestion level: Low, Medium, or High")
    confidence: Optional[float] = Field(None, description="Prediction probability confidence score (0.0 - 1.0)")
    recommendation: str = Field(..., description="Actionable traffic management recommendation")


class AccidentPredictionResponse(BaseModel):
    """Schema for /predict/accident response."""
    risk_level: str = Field(..., description="Estimated accident risk level: Low, Moderate, or High")
    risk_percentage: Optional[float] = Field(None, description="Estimated confidence percentage (0.0% - 100.0%)")
    recommendation: str = Field(..., description="Actionable safety guidance recommendation")
    disclaimer: str = Field(
        default="This prediction is an AI-based ESTIMATED ACCIDENT RISK for decision-support and does not guarantee accident occurrence.",
        description="Mandatory disclaimer clarifying estimated risk vs actual accident occurrence"
    )


class TrafficFeaturesInput(BaseModel):
    """
    Input schema supporting both friendly categorical names and exact raw feature values.
    extra='allow' enables passing any of the 73 one-hot columns directly.
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    # Core environmental and numerical features
    temp: float = Field(default=288.15, description="Temperature in Kelvin (e.g. 288.15 K = 15°C)")
    rain_1h: float = Field(default=0.0, description="Rainfall volume in the last hour in mm")
    snow_1h: float = Field(default=0.0, description="Snowfall volume in the last hour in mm")
    clouds_all: int = Field(default=40, ge=0, le=100, description="Cloudiness percentage (0-100%)")
    hour: int = Field(default=12, ge=0, le=23, description="Hour of the day (0-23)")
    day: int = Field(default=1, ge=1, le=31, description="Day of the month (1-31)")
    month: int = Field(default=1, ge=1, le=12, description="Month of the year (1-12)")
    year: int = Field(default=2024, description="Year (e.g. 2024)")
    day_of_week: int = Field(default=0, ge=0, le=6, description="Day of the week (0=Monday, 6=Sunday)")
    is_weekend: int = Field(default=0, ge=0, le=1, description="1 if weekend (Sat/Sun), else 0")
    is_peak_hour: int = Field(default=0, ge=0, le=1, description="1 if peak rush hour (e.g. 7-9 AM, 4-7 PM), else 0")
    
    # Engineered risk indices
    traffic_risk: float = Field(default=0.0, description="Calculated traffic risk index")
    rain_risk: float = Field(default=0.0, description="Calculated rain risk index")
    snow_risk: float = Field(default=0.0, description="Calculated snow risk index")
    weather_risk: float = Field(default=0.0, description="Combined weather risk index")

    # High-level friendly categorical inputs (automatically converted to one-hot columns)
    holiday: Optional[str] = Field(default="None", description="Holiday name (e.g. 'None', 'Labor Day', 'Thanksgiving Day')")
    weather_main: Optional[str] = Field(default="Clouds", description="General weather category (e.g. 'Clouds', 'Rain', 'Clear')")
    weather_description: Optional[str] = Field(default="scattered clouds", description="Detailed weather description (e.g. 'broken clouds', 'light rain')")


class CongestionInputSchema(TrafficFeaturesInput):
    """Request schema for congestion prediction endpoint."""
    pass


class AccidentInputSchema(TrafficFeaturesInput):
    """Request schema for accident risk estimation endpoint."""
    pass


class ErrorResponse(BaseModel):
    """Standardized API error response schema."""
    detail: str = Field(..., description="Human-readable error explanation")
    error_type: str = Field(..., description="Error classification identifier")
