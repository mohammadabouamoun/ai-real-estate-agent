import streamlit as st
import requests

# Set to True when deploying to Streamlit Cloud, False for local testing
USE_LIVE_API = True   # change to False when running locally

if USE_LIVE_API:
    API_BASE = "https://ai-real-estate-agent-1vr4.onrender.com"
else:
    API_BASE = "http://localhost:8000"

st.set_page_config(page_title="AI Real Estate Agent", layout="centered")
st.title("🏠 AI Real Estate Agent")

# Initialize session state
if "extracted_features" not in st.session_state:
    st.session_state.extracted_features = None
if "missing_fields" not in st.session_state:
    st.session_state.missing_fields = []
if "complete_features" not in st.session_state:
    st.session_state.complete_features = None

query = st.text_area("Describe the property:", "3-bedroom ranch with a big garage")
prompt_version = st.selectbox("Prompt version", ["v1", "v2"])

if st.button("Extract Features"):
    with st.spinner("Extracting features..."):
        try:
            resp = requests.post(
                f"{API_BASE}/extract",
                params={"query": query, "prompt_version": prompt_version}
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.extracted_features = data["extracted"]
                st.session_state.missing_fields = data["missing_fields"]
                st.success("Features extracted. Fill in the missing fields below.")
            else:
                st.error(f"Extraction failed: {resp.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

# If we have extracted features, show them and allow filling gaps
if st.session_state.extracted_features:
    st.subheader("Extracted Features")
    # Display non‑missing values read‑only
    for key, val in st.session_state.extracted_features.items():
        if key not in st.session_state.missing_fields:
            st.text(f"{key}: {val}")
    
    st.subheader("Provide Missing Information")
    filled_features = st.session_state.extracted_features.copy()
    
    # Create input widgets for each missing field (no defaults, min_value=0 for numeric)
    for field in st.session_state.missing_fields:
        if field in ["LotArea", "GarageCars", "BedroomAbvGr", "FullBath", "HalfBath", "OverallQual", "TotRmsAbvGrd"]:
            # Numeric fields: prevent negative numbers
            val = st.number_input(f"{field}", value=None, step=1, min_value=0, placeholder=f"Enter {field}")
            filled_features[field] = val
        elif field == "Neighborhood":
            val = st.text_input(f"{field}", placeholder="Enter neighborhood (e.g., NAmes, CollgCr)")
            filled_features[field] = val if val else None
        elif field == "KitchenQual":
            val = st.selectbox(f"{field}", ["", "Po", "Fa", "TA", "Gd", "Ex"], index=0)
            filled_features[field] = val if val else None
        elif field ==  "YearBuilt" :
            val = st.number_input(f"{field}", value=None, step=1, min_value=1872, max_value=2010, placeholder=f"Enter {field}")
            filled_features[field] = val
    # Prediction button – only works when all missing fields are filled
    if st.button("Get Price Prediction"):
        # Check that all missing fields have been filled
        still_missing = [f for f in st.session_state.missing_fields if filled_features.get(f) in (None, "")]
        if still_missing:
            st.error(f"Please fill in the missing fields: {still_missing}")
        else:
            # All fields are filled – proceed
            complete_features = {k: v for k, v in filled_features.items()}
            with st.spinner("Predicting..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/predict",
                        params={"query": query, "prompt_version": prompt_version},
                        json=complete_features
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.subheader(f"Predicted Price: ${data['predicted_price']:,.0f}")
                        st.subheader("Interpretation")
                        st.write(data["interpretation"])
                    else:
                        st.error(f"Prediction failed: {resp.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")