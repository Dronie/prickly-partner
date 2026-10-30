# This file should handle HTTP requests and the service lifecycle.
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from backend.inference import (
    InferenceService,
    ModelUnavailableError,
)

logger = logging.getLogger(__name__)

ObservationValue = Annotated[
    int,
    Field(strict=True, ge=0, le=1),
]

AgentId = Annotated[
    int,
    Field(strict=True, ge=0, le=9),
]

ActionId = Annotated[
    int,
    Field(strict=True, ge=0, le=4),
]

class ActionRequest(BaseModel):
    observation: list[ObservationValue] = Field(
        min_length=100,
        max_length=100,
    )
    agent: Literal["red", "blue"]
    agent_id: AgentId
    opponent_id: AgentId

class ActionResponse(BaseModel):
    action: ActionId


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference = InferenceService()

    try:
        yield
    finally:
        app.state.inference.sessions.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/api/action", response_model=ActionResponse)
def get_action(body: ActionRequest, request: Request):
    service = request.app.state.inference

    try:
        action = service.choose_action(
            observation=body.observation,
            agent_id=body.agent_id,
            opponent_id=body.opponent_id,
        )
    except ModelUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error
    except Exception as error:
        logger.exception("Model inference failed.")

        raise HTTPException(
            status_code=500,
            detail="Model inference failed.",
        ) from error

    return ActionResponse(action=action)

@app.get("/health")
def health(request: Request):
    loaded_models = len(request.app.state.inference.sessions)

    if loaded_models != 10:
        raise HTTPException(
            status_code = 503,
            detail = "Not all models are loaded."
        )

    return {
        "status": "ready",
        "loaded_models": loaded_models
    }
