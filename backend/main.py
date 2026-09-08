import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    HealthResponse,
    CongestionInputSchema,
    CongestionPredictionResponse,
    AccidentInputSchema,
    AccidentPredictionResponse,
)
from backend.predictor import (
    get_predictor,
    ModelLoadError,
    PredictionExecutionError,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ai_traffic_system")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler to load ML models once at startup.
    """
    logger.info("Initializing AI Traffic Risk Prediction System backend...")
    try:
        predictor = get_predictor()
        logger.info("Predictor initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize models during startup: {e}")
    yield
    logger.info("Shutting down AI Traffic Risk Prediction System backend...")

app = FastAPI(
    title="AI-Based Traffic Congestion & Accident Risk Prediction System",
    description=(
        "Decision-support system providing AI-based traffic congestion analysis and "
        "accident risk estimation using pre-trained machine learning models."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler to hide internal tracebacks from API clients
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the prediction request."}
    )

@app.exception_handler(ModelLoadError)
async def model_load_exception_handler(request: Request, exc: ModelLoadError):
    logger.error(f"Model loading error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"Model service unavailable: {str(exc)}"}
    )

@app.exception_handler(PredictionExecutionError)
async def prediction_error_handler(request: Request, exc: PredictionExecutionError):
    logger.error(f"Prediction execution error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": f"Prediction failed: {str(exc)}"}
    )

@app.get(
    "/",
    summary="Root",
    tags=["System"]
)
async def root():
    return {
        "message": "AI-Based Traffic Congestion & Accident Risk Prediction API is running.",
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["System"]
)
async def health_check():
    """
    Health check endpoint returning system operational status.
    """
    return {"status": "healthy"}

@app.post(
    "/predict/congestion",
    response_model=CongestionPredictionResponse,
    summary="Predict Traffic Congestion",
    tags=["Prediction"]
)
async def predict_congestion(payload: CongestionInputSchema):
    """
    Accepts traffic and environmental features, aligns them with the trained model's feature schema,
    and returns the predicted congestion level, confidence score, and tailored recommendation.
    """
    try:
        predictor = get_predictor()
        # Convert schema to dict using aliases to preserve exact feature names
        input_dict = payload.model_dump(by_alias=True)
        result = predictor.predict_congestion(input_dict)
        return result
    except (ModelLoadError, PredictionExecutionError) as e:
        raise
    except Exception as e:
        logger.error(f"Error in congestion prediction route: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing traffic congestion prediction."
        )

@app.post(
    "/predict/accident",
    response_model=AccidentPredictionResponse,
    summary="Predict Accident Risk",
    tags=["Prediction"]
)
async def predict_accident(payload: AccidentInputSchema):
    """
    Accepts traffic and environmental features, aligns them with the trained model's feature schema,
    and returns an AI-based accident risk estimation (Low, Moderate, High), risk percentage,
    and safety recommendation.
    """
    try:
        predictor = get_predictor()
        # Convert schema to dict using aliases to preserve exact feature names
        input_dict = payload.model_dump(by_alias=True)
        result = predictor.predict_accident_risk(input_dict)
        return result
    except (ModelLoadError, PredictionExecutionError) as e:
        raise
    except Exception as e:
        logger.error(f"Error in accident prediction route: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing accident risk prediction."
        )
