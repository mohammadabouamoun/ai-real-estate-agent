from pydantic import BaseModel
from typing import Optional, List

class ExtractedFeatures(BaseModel):
    """Features extracted by LLM Stage 1."""
    LotArea: Optional[int] = None
    BedroomAbvGr: Optional[int] = None
    FullBath: Optional[int] = None
    HalfBath: Optional[int] = None
    Neighborhood: Optional[str] = None
    OverallQual: Optional[int] = None
    YearBuilt: Optional[int] = None
    GarageCars: Optional[int] = None
    KitchenQual: Optional[str] = None
    TotRmsAbvGrd: Optional[int] = None

class PredictionResponse(BaseModel):
    """Final API response."""
    query: str
    extracted_features: ExtractedFeatures
    predicted_price: float
    interpretation: str