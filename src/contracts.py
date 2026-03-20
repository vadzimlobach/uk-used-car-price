from collections.abc import Sequence
from typing import Protocol

import pandas as pd
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_price: float
    model_run_id: str


class HealthResponse(BaseModel):
    status: str


class SupportsPredict(Protocol):
    def predict(self, X: pd.DataFrame) -> Sequence[float]: ...
