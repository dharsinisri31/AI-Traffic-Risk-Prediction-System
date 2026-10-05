"""
AI Traffic Risk Prediction System - FastAPI Application Entry Point

This file initializes the FastAPI application, mounts CORS middleware,
manages model preloading on startup, registers route handlers,
and configures global exception handling.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas import (
    HealthResponse,
    CongestionInputSchema,
    CongestionPredictionResponse,
    AccidentInputSchema,
    AccidentPredictionResponse,
    ErrorResponse,
)
from backend.predictor import (
    get_predictor,
    ModelArtifactMissingError,
    ModelLoadError,
    PredictionExecutionError,
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ai_traffic_system.api")


# Application lifespan context manager to load models at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler that loads models into memory on server startup
    and handles graceful shutdown.
    """
    logger.info("Initializing AI Traffic Risk Prediction System...")
    try:
        predictor = get_predictor()
        logger.info(
            f"API ready with {len(predictor.congestion_features)} congestion features "
            f"and {len(predictor.accident_features)} accident features."
        )
    except Exception as e:
        logger.critical(f"Startup model load failed: {e}", exc_info=True)
    yield
    logger.info("Shutting down AI Traffic Risk Prediction System.")


# Create FastAPI application
app = FastAPI(
    title="AI Traffic Risk Prediction System API",
    description=(
        "REST API powered by Machine Learning for real-time traffic congestion prediction "
        "and estimated accident risk assessment based on meteorological and temporal data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handlers
@app.exception_handler(ModelArtifactMissingError)
async def model_missing_exception_handler(request, exc: ModelArtifactMissingError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "error_type": "ModelArtifactMissingError"},
    )


@app.exception_handler(ModelLoadError)
async def model_load_exception_handler(request, exc: ModelLoadError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "error_type": "ModelLoadError"},
    )


@app.exception_handler(PredictionExecutionError)
async def prediction_execution_exception_handler(request, exc: PredictionExecutionError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_type": "PredictionExecutionError"},
    )


# Root Endpoint
@app.get("/", summary="Root index", tags=["System"])
async def root():
    """Returns a welcome message with a link to interactive documentation."""
    return {
        "message": "Welcome to the AI Traffic Risk Prediction System API",
        "docs_url": "/docs",
        "health_check": "/health",
    }


# Health Check Endpoint
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    tags=["System"],
)
async def health_check():
    """
    Verifies that the API service is alive and that the ML models and feature column
    artifacts are successfully loaded in memory.
    """
    try:
        predictor = get_predictor()
        models_ready = (
            predictor.congestion_model is not None and
            predictor.accident_model is not None
        )
        return HealthResponse(
            status="healthy" if models_ready else "degraded",
            service="AI Traffic Risk Prediction System API",
            models_loaded=models_ready,
            congestion_features_count=len(predictor.congestion_features),
            accident_features_count=len(predictor.accident_features),
        )
    except Exception as e:
        logger.error(f"Health check encountered error: {e}")
        return HealthResponse(
            status="unhealthy",
            service="AI Traffic Risk Prediction System API",
            models_loaded=False,
            congestion_features_count=0,
            accident_features_count=0,
        )


# Congestion Prediction Endpoint
@app.post(
    "/predict/congestion",
    response_model=CongestionPredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Predict Traffic Congestion Level",
    tags=["Predictions"],
)
async def predict_congestion(payload: CongestionInputSchema):
    """
    Receives traffic and weather parameters, aligns them with the trained 73-feature columns,
    and predicts the traffic congestion level (`Low`, `Medium`, or `High`).
    """
    try:
        predictor = get_predictor()
        input_data = payload.model_dump()
        result = predictor.predict_congestion(input_data)
        return CongestionPredictionResponse(**result)
    except (ModelArtifactMissingError, ModelLoadError) as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except PredictionExecutionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error in /predict/congestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected prediction error occurred: {str(e)}",
        )


# Accident Risk Prediction Endpoint
@app.post(
    "/predict/accident",
    response_model=AccidentPredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Predict Estimated Accident Risk",
    tags=["Predictions"],
)
async def predict_accident(payload: AccidentInputSchema):
    """
    Receives traffic and weather parameters, aligns them with the trained 73-feature columns,
    and returns an **ESTIMATED ACCIDENT RISK** (`Low`, `Moderate`, or `High`).

    **Notice**: The training dataset does not contain verified real accident labels;
    thus, this output is strictly an estimated risk metric for decision support.
    """
    try:
        predictor = get_predictor()
        input_data = payload.model_dump()
        result = predictor.predict_accident_risk(input_data)
        return AccidentPredictionResponse(**result)
    except (ModelArtifactMissingError, ModelLoadError) as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except PredictionExecutionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error in /predict/accident: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected prediction error occurred: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
