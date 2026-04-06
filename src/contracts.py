from collections.abc import Sequence
from typing import Protocol

import pandas as pd
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class ModelVersion(BaseModel):
    run_id: str = Field(..., description="Artifact run identifier")
    git_commit: str = Field(..., description="Git commit hash used to build/train the model")
    model_type: str | None = Field(default=None, description="Model family, e.g. 'rf'")


class PredictionResponse(BaseModel):
    predicted_price: float
    model_version: ModelVersion


class MetadataResponse(BaseModel):
    service_name: str
    model_version: ModelVersion
    schema_version: str
    prediction_features: list[str]


class SupportsPredict(Protocol):
    def predict(self, X: pd.DataFrame) -> Sequence[float]: ...
