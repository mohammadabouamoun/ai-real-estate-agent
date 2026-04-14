from fastapi import FastAPI, HTTPException
from src.schemas import ExtractedFeatures, PredictionResponse
from src.llm_client import extract_features_stage1, interpret_prediction_stage2
import pickle
import pandas as pd
import numpy as np
import os

app = FastAPI(title="AI Real Estate Agent")

# Load model and imputation parameters at startup
MODEL_PATH = 'models/best_model.pkl'
IMPUTER_PATH = 'processed/imputation_params.pkl'
Y_TRAIN_PATH = 'processed/y_train.csv'

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
if not os.path.exists(IMPUTER_PATH):
    raise RuntimeError(f"Imputation params not found at {IMPUTER_PATH}. Run data_cleaning.py first.")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(IMPUTER_PATH, 'rb') as f:
    imputers = pickle.load(f)

train_prices = pd.read_csv(Y_TRAIN_PATH).squeeze()
median_price = train_prices.median()
price_min = train_prices.min()
price_max = train_prices.max()

def apply_imputation_to_input(features: ExtractedFeatures) -> pd.DataFrame:
    """Convert extracted features to DataFrame and apply same imputation as training."""
    df = pd.DataFrame([features.dict()])
     # Replace Python None with np.nan (so SimpleImputer can handle them)
    df = df.replace({None: np.nan})
    # GarageCars: missing -> 0
    if df['GarageCars'].isnull().any():
        df['GarageCars'] = df['GarageCars'].fillna(0)
    # Numeric median columns
    for col in imputers['numeric_median_cols']:
        if df[col].isnull().any():
            imp = imputers[f'median_{col}']
            df[col] = imp.transform(df[[col]]).ravel()
    # Categorical mode columns
    for col in imputers['categorical_mode_cols']:
        if df[col].isnull().any():
            imp = imputers[f'mode_{col}']
            df[col] = imp.transform(df[[col]]).ravel()
    return df

@app.post("/predict", response_model=PredictionResponse)
async def predict(query: str, prompt_version: str = "v1"):
    # Stage 1: LLM extraction
    try:
        extracted_dict = extract_features_stage1(query, version=prompt_version)
        extracted = ExtractedFeatures(**extracted_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Stage 1 extraction failed: {str(e)}")
    
    # Prepare input for model
    input_df = apply_imputation_to_input(extracted)
    # Model prediction
    try:
        pred = model.predict(input_df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
    
    # Stage 2: interpretation (pass extracted features)
    try:
        interpretation = interpret_prediction_stage2(query, pred, median_price, (price_min, price_max), extracted.dict())
    except Exception as e:
        interpretation = f"Predicted price: ${pred:,.0f}. (Interpretation unavailable: {str(e)})"
    
    return PredictionResponse(
        query=query,
        extracted_features=extracted,
        predicted_price=float(pred),
        interpretation=interpretation
    )

@app.get("/health")
def health():
    return {"status": "ok"}
