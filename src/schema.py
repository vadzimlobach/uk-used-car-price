from typing import List
from pydantic import BaseModel, Field, ConfigDict

RAW_FEATURES: List[str] = [
    "brand",
    "model", 
    "year", 
    "price", 
    "transmission",
    "mileage",
    "fuelType",
    "tax",
    "mpg",
    "engineSize",
]

class CarFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    year: int = Field(..., ge=1950, le=2026, description="Year of manufacture")
    mileage: int = Field(..., ge=0, description="Mileage in miles")
    tax: float = Field(..., ge=0,description="Tax in pounds")
    mpg: float = Field(..., ge=0, description="Miles per gallon")
    engineSize: float = Field(..., ge=0, description="Engine size in liters")
    brand: str = Field(..., min_length=1,description="Car brand")
    model: str = Field(..., min_length=1,description="Car model")
    transmission: str = Field(..., min_length=1,description="Transmission type (e.g. manual, automatic)")
    fuelType: str = Field(..., min_length=1, description="Fuel type (e.g. petrol, diesel)")

    def to_dict(self) -> dict:
        return self.model_dump()

class CarTrainingFeatures(CarFeatures):
    price: int = Field(..., ge=0, description="Price in pounds")