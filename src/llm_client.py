import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # loads GROQ_API_KEY from .env file

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "llama-3.3-70b-versatile"

def extract_features_stage1(query: str, version: str = "v1") -> dict:
    """Extract features using Groq API."""
    if version == "v1":
        system_prompt = """You are a real estate data extractor. From the user's description, extract the following features as JSON.
If a feature is not mentioned, set its value to null.
Features: LotArea (sq ft, integer), BedroomAbvGr (integer), FullBath (integer), HalfBath (integer),
Neighborhood (string), OverallQual (1-10 integer), YearBuilt (integer),
GarageCars (integer, 0 if no garage), KitchenQual (string: Po, Fa, TA, Gd, Ex),
TotRmsAbvGrd (integer).

Return ONLY valid JSON with these keys. No extra text."""
    else:  # v2 – shorter prompt
        system_prompt = """From the text, output JSON with these keys: LotArea, BedroomAbvGr, FullBath, HalfBath, Neighborhood, OverallQual, YearBuilt, GarageCars, KitchenQual, TotRmsAbvGrd. Use null if missing."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            model=MODEL_NAME,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"Extraction error: {e}")
        return {}

def interpret_prediction_stage2(query: str, predicted_price: float,
                                median_price: float, price_range: tuple,
                                extracted_features: dict) -> str:
    """Generate short interpretation using Groq."""
    # Build a concise features string (skip None values)
    feat_items = [f"{k}:{v}" for k, v in extracted_features.items() if v is not None]
    feat_str = ", ".join(feat_items) if feat_items else "no specific features"
    user_prompt = f"""User: "{query}"
Features: {feat_str}
Predicted price: ${predicted_price:,.0f}. Median: ${median_price:,.0f}. Range: ${price_range[0]:,.0f}-${price_range[1]:,.0f}.
Explain in 1-2 sentences if price is high/low and why. Be brief."""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful real estate AI assistant."},
                {"role": "user", "content": user_prompt}
            ],
            model=MODEL_NAME,
            temperature=0.5,
            max_tokens=80
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Predicted price: ${predicted_price:,.0f}. (Interpretation unavailable: {str(e)})"