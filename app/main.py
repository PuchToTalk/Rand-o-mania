"""
FastAPI server for Rand-o-mania.

Exposes POST /execute endpoint for processing natural language mathematical operations.
"""

import logging
import os
import random
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schema import (
    ExecuteRequest,
    ExecuteResponse,
    ErrorResponse,
    ErrorDetail,
    ExecutePlanRequest,
    ExecutionPlan,
    DebugInfo,
    DebugStats,
    DebugLimits,
)
from app.llm_processor import LLMProcessor, LLMProcessingError
from app.interpreter import (
    Interpreter,
    ExecutionError,
    DivisionByZeroError,
    NegativeSqrtError,
    InvalidOperationError,
    UndefinedVariableError,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global LLM processor instance
llm_processor: Optional[LLMProcessor] = None


def create_error_response(code: str, message: str, status_code: int) -> JSONResponse:
    """
    Create a structured error response.
    
    Args:
        code: Error code (e.g., "DIVISION_BY_ZERO")
        message: Human-readable error message
        status_code: HTTP status code
        
    Returns:
        JSONResponse with structured error
    """
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}}
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global llm_processor
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set - /execute endpoint will fail")
    else:
        try:
            llm_processor = LLMProcessor(api_key=api_key)
            logger.info("LLM processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM processor: {e}")
    
    yield
    
    logger.info("Shutting down")


app = FastAPI(
    title="Rand-o-mania",
    description="Execute mathematical operations on random numbers via natural language",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_plan(
    plan: ExecutionPlan,
    seed: Optional[int] = None,
    debug: bool = False
) -> ExecuteResponse:
    """
    Shared execution logic for both /execute and /execute_plan endpoints.
    
    Args:
        plan: Validated execution plan
        seed: Optional seed for deterministic RNG
        debug: Whether to include debug information
        
    Returns:
        ExecuteResponse with result and optional debug info
        
    Raises:
        Various ExecutionError subclasses
    """
    # Create RNG with seed if provided
    if seed is not None:
        rng_instance = random.Random(seed)
        rng = rng_instance.random
    else:
        rng = None
    
    # Execute plan
    interpreter = Interpreter(rng=rng, debug=debug)
    result, random_numbers, debug_info = interpreter.execute(plan)
    
    # Build response
    response_data = {
        "result": result,
        "random_integers": random_numbers
    }
    
    # Add debug info if requested
    if debug and debug_info:
        response_data["debug"] = DebugInfo(
            plan=plan.model_dump(),
            execution_trace=debug_info["execution_trace"],
            final_vars=debug_info["final_vars"],
            result_str=debug_info["result_str"],
            seed_used=seed,
            stats=DebugStats(
                steps_executed=debug_info["steps_executed"],
                random_count=debug_info["random_count"],
                max_depth_reached=debug_info["max_depth_reached"]
            ),
            limits=DebugLimits(
                max_steps=Interpreter.MAX_STEPS,
                max_depth=Interpreter.MAX_DEPTH
            )
        )
    
    return ExecuteResponse(**response_data)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "llm_ready": llm_processor is not None}


@app.post(
    "/execute", 
    response_model=ExecuteResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid prompt or plan"},
        422: {"model": ErrorResponse, "description": "Execution error (div by zero, sqrt negative, etc.)"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def execute(request: ExecuteRequest):
    """
    Execute mathematical operations based on natural language prompt.
    
    The prompt is converted to an execution plan by the LLM (with retry on validation failure),
    then executed by a safe interpreter using Decimal arithmetic.
    
    Returns the result rounded to 10 decimal places and all random numbers generated.
    
    Note: "random_integers" contains floats in [0,1] despite the field name (kept for API compatibility).
    """
    if llm_processor is None:
        return create_error_response(
            code="LLM_NOT_INITIALIZED",
            message="LLM processor not initialized - check OPENAI_API_KEY environment variable",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    logger.info(f"Processing prompt: {request.prompt[:100]}...")
    
    # Step 1: Convert prompt to execution plan via LLM (with retry logic)
    try:
        plan = llm_processor.process(request.prompt)
        logger.info(f"Generated execution plan with {len(plan.steps)} steps")
    except LLMProcessingError as e:
        logger.error(f"LLM processing error: {e}")
        return create_error_response(
            code="LLM_PROCESSING_ERROR",
            message=f"Failed to process prompt: {str(e)}",
            status_code=status.HTTP_400_BAD_REQUEST
        )
    
    # Step 2: Execute the plan safely
    try:
        response = run_plan(plan, seed=request.seed, debug=request.debug)
        logger.info(f"Execution complete: result={response.result}, randoms={len(response.random_integers)}")
        return response
    except DivisionByZeroError as e:
        logger.error(f"Division by zero: {e}")
        return create_error_response(
            code="DIVISION_BY_ZERO",
            message=str(e),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except NegativeSqrtError as e:
        logger.error(f"Negative sqrt: {e}")
        return create_error_response(
            code="NEGATIVE_SQRT",
            message=str(e),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except UndefinedVariableError as e:
        logger.error(f"Undefined variable: {e}")
        return create_error_response(
            code="UNDEFINED_VARIABLE",
            message=str(e),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except InvalidOperationError as e:
        logger.error(f"Invalid operation: {e}")
        return create_error_response(
            code="INVALID_OPERATION",
            message=str(e),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except ExecutionError as e:
        logger.error(f"Execution error: {e}")
        return create_error_response(
            code="EXECUTION_ERROR",
            message=str(e),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except Exception as e:
        logger.exception(f"Unexpected error during execution: {e}")
        return create_error_response(
            code="INTERNAL_ERROR",
            message=f"Unexpected error: {str(e)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.post(
    "/execute_plan",
    response_model=ExecuteResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Execution error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def execute_plan(request: ExecutePlanRequest):
    """
    Execute a plan directly without LLM processing.
    
    Useful for testing and integrations where you already have a structured plan.
    Bypasses OpenAI API completely.
    
    Args:
        request: Contains the execution plan, optional seed, and debug flag
        
    Returns:
        ExecuteResponse with result and optional debug info
    """
    logger.info(f"Executing plan directly with {len(request.plan.steps)} steps")
    
    try:
        response = run_plan(request.plan, seed=request.seed, debug=request.debug)
        logger.info(f"Direct execution complete: result={response.result}")
        return response
    except DivisionByZeroError as e:
        logger.error(f"Division by zero: {e}")
        return create_error_response(
            code="DIVISION_BY_ZERO",
            message=str(e),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except NegativeSqrtError as e:
        logger.error(f"Negative sqrt: {e}")
        return create_error_response(
            code="NEGATIVE_SQRT",
            message=str(e),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except UndefinedVariableError as e:
        logger.error(f"Undefined variable: {e}")
        return create_error_response(
            code="UNDEFINED_VARIABLE",
            message=str(e),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except InvalidOperationError as e:
        logger.error(f"Invalid operation: {e}")
        return create_error_response(
            code="INVALID_OPERATION",
            message=str(e),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except ExecutionError as e:
        logger.error(f"Execution error: {e}")
        return create_error_response(
            code="EXECUTION_ERROR",
            message=str(e),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except Exception as e:
        logger.exception(f"Unexpected error during execution: {e}")
        return create_error_response(
            code="INTERNAL_ERROR",
            message=f"Unexpected error: {str(e)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

