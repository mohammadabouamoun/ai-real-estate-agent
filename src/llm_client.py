import json
import requests


MODEL_NAME = "llama3.2:latest"  
OLLAMA_URL = "http://localhost:11434/api/generate"

def extract_features_stage1(query: str, version: str = "v1") -> dict:
    """Extract features using short, efficient prompts."""
    if version == "v1":
        prompt = f"""Extract house features from this description as JSON. Use null if missing.
Features: LotArea (int sqft), BedroomAbvGr (int), FullBath (int), HalfBath (int),
Neighborhood (str), OverallQual (1-10 int), YearBuilt (int),
GarageCars (int, 0 if none), KitchenQual (Po/Fa/TA/Gd/Ex), TotRmsAbvGrd (int).
Query: "{query}"
Return ONLY valid JSON."""
    else:  # v2 – even shorter
        prompt = f"""From the text, output JSON with these keys: LotArea, BedroomAbvGr, FullBath, HalfBath, Neighborhood, OverallQual, YearBuilt, GarageCars, KitchenQual, TotRmsAbvGrd. Use null if missing.
Text: "{query}"
JSON:"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,       # low temp for deterministic extraction
                    "num_predict": 256        # limit output length
                }
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return json.loads(result["response"])
    except Exception as e:
        print(f"LLM extraction error: {e}")
        return {}

def interpret_prediction_stage2(query: str, predicted_price: float,
                                median_price: float, price_range: tuple,
                                extracted_features: dict) -> str:
    """Short, fast interpretation prompt."""
    # Build a concise features string (skip None)
    feat_items = [f"{k}:{v}" for k, v in extracted_features.items() if v is not None]
    feat_str = ", ".join(feat_items) if feat_items else "no specific features"
    prompt = f"""User: "{query}"
Features: {feat_str}
Predicted price: ${predicted_price:,.0f}. Median: ${median_price:,.0f}. Range: ${price_range[0]:,.0f}-${price_range[1]:,.0f}.

Explain in 1-2 sentences if price is high/low and why. Be brief."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "num_predict": 80         # very short output
                }
            },
            timeout=20
        )
        response.raise_for_status()
        result = response.json()
        return result["response"].strip()
    except Exception as e:
        return f"Predicted price: ${predicted_price:,.0f}. (Interpretation unavailable: {str(e)})"