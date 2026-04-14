import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:latest"

def extract_features_stage1(query: str, version: str = "v1") -> dict:
    """Extract features from natural language using Ollama."""
    if version == "v1":
        prompt = f"""
You are a real estate data extractor. From the user's description, extract the following features as JSON.
If a feature is not mentioned, set its value to null.
Features: LotArea (sq ft, integer), BedroomAbvGr (integer), FullBath (integer), HalfBath (integer),
Neighborhood (string), OverallQual (1-10 integer), YearBuilt (integer),
GarageCars (integer, 0 if no garage), KitchenQual (string: Po, Fa, TA, Gd, Ex),
TotRmsAbvGrd (integer).

Return ONLY valid JSON with these keys. No extra text.
User query: "{query}"
"""
    else:  # v2 – more conversational
        prompt = f"""
Extract house features from this description. Use null for missing.
Features: lot area (sq ft), bedrooms, full baths, half baths, neighborhood,
overall quality (1-10), year built, garage capacity (0 if none), kitchen quality (Po/Fa/TA/Gd/Ex),
total rooms above grade.
Return ONLY valid JSON.
Query: "{query}"
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "format": "json"
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
    """Generate detailed interpretation using extracted features."""
    # Format the extracted features as a readable string (skip None values)
    features_str = "\n".join([f"  - {k}: {v}" for k, v in extracted_features.items() if v is not None])
    if not features_str:
        features_str = "  (no specific features were extracted)"

    prompt = f"""
You are a real estate AI. The user asked: "{query}"

From that description, the following features were extracted:
{features_str}

Our model predicted a price of ${predicted_price:,.0f}.
The median sale price in our training data is ${median_price:,.0f}, and typical prices range from ${price_range[0]:,.0f} to ${price_range[1]:,.0f}.

Now, provide a helpful, detailed explanation to the user:
- Compare the predicted price to the median (is it high, low, or average?).
- Explain which specific features (e.g., large lot, many bedrooms, high quality, good neighborhood, garage, etc.) likely drove the price up or down.
- Give a short summary (3-5 sentences) that answers "Why this price?".

Be conversational and informative.
"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "max_tokens": 300}
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result["response"].strip()
    except Exception as e:
        return f"Predicted price: ${predicted_price:,.0f}. (Detailed interpretation unavailable: {str(e)})"