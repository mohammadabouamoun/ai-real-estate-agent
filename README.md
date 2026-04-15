# 🏠 AI Real Estate Agent

An end-to-end AI system that extracts property features from natural language, predicts house prices using a trained ML model, and explains the prediction in plain English. Built with **FastAPI**, **Streamlit**, **Docker**, and **Groq LLM API** (or local Ollama).  


**Key features**:
- Two‑stage LLM prompt chain:  
  - Stage 1: Extract structured features from a user’s description.  
  - Stage 2: Interpret the predicted price using the extracted features.  
- Scikit‑learn ML pipeline (imputation, encoding, scaling, Random Forest).  
- No silent defaults – users review and fill missing features before prediction.  
- REST API (FastAPI) and interactive UI (Streamlit).  
- Containerised with Docker.  
- Prompt versioning experiment (two variants, winner `v1`).  

---

## 📁 Repository Structure
ai-real-estate-agent/
├── src/
│ ├── data_cleaning.py # Split, imputation, save preprocessed data
│ ├── train_model.py # Encode, scale, train two models, save best
│ ├── schemas.py # Pydantic models for features & response
│ ├── llm_client.py # Groq API calls (Stage 1 & 2)
│ ├── api.py # FastAPI endpoints: /extract, /predict, /health
│ └── ui.py # Streamlit UI – gap‑filling before prediction
├── models/
│ └── best_model.pkl # Serialised Random Forest pipeline
├── processed/
│ └── imputation_params.pkl # Fitted imputers (median, mode, zero)
├── requirements.txt # Production dependencies
├── requirements-dev.txt # Development extras (jupyter, black, pytest)
├── Dockerfile # Container definition
├── .env.example # Template for GROQ_API_KEY
├── experiment_prompts.py # Prompt versioning test script
└── README.md # This file

## 📓 Colab Notebook

The full EDA, ML pipeline, and prompt versioning experiment are also available as a Google Colab notebook.  
**Link**: [AI Real Estate Agent – Colab Notebook](https://colab.research.google.com/drive/1NvLJ-76SyOG1bW7EgQPG2JcUqW6HY51S?usp=sharing)
 
 Note: The Colab notebook uses the OpenML version of the dataset (with underscore column names) and a slightly different ordinal encoding for KitchenQual. The local pipeline uses a CSV file with camelCase names. Both produce similar results (R² ~0.82–0.83).

## 🐳 Docker Deployment

The FastAPI app is containerized using Docker and deployed on Render.

- **Dockerfile**: [link to Dockerfile](Dockerfile)
- **Live API URL**: [https://ai-real-estate-agent-1vr4.onrender.com](https://ai-real-estate-agent-1vr4.onrender.com)
- **Build command**: `docker build -t ai-real-estate-agent .`
- **Run command**: `docker run -p 8000:8000 --env-file .env ai-real-estate-agent`

The deployment is automatically triggered on each push to the `feature/ml-pipeline` branch. The service runs the container with the environment variable `GROQ_API_KEY` set for LLM calls.



text

---

## 🧠 Architecture

The system implements a **prompt chain**:  

**User query** → **Stage 1 (LLM)** → **Extracted features** → **ML prediction** → **Stage 2 (LLM)** → **Interpretation** → **Response**

- **Stage 1**: The LLM (Groq `llama-3.3-70b-versatile`) extracts up to 10 features from the natural language query. Missing features are reported (not imputed silently).  
- **ML Pipeline**: A Random Forest model (trained on the full Ames Housing dataset, 2930 rows) predicts the sale price. The pipeline includes median imputation, standard scaling, ordinal encoding for `KitchenQual`, and one‑hot encoding for `Neighborhood`.  
- **Stage 2**: The LLM receives the extracted features, the predicted price, and training statistics (median price, range) to produce a user‑friendly explanation.  
- **UI (Streamlit)** : Shows extracted features, lets the user fill any missing fields, then requests the prediction.  
- **API (FastAPI)** : Serves the `/extract` and `/predict` endpoints. Loads the model and imputers at startup.  
- **Docker**: Packages the FastAPI app for easy deployment.

---

## 📊 Dataset & Preprocessing

- **Dataset**: Full Ames Housing dataset (2,930 rows, 82 columns) – loaded from a local CSV or OpenML (`data_id=41211`).  
- **Target**: `SalePrice`.  
- **Selected features** (10):  
  `LotArea`, `BedroomAbvGr`, `FullBath`, `HalfBath`, `Neighborhood`, `OverallQual`, `YearBuilt`, `GarageCars`, `KitchenQual`, `TotRmsAbvGrd`.  

**Preprocessing** (no leakage):
1. Train/validation/test split: 70/15/15 with `random_state=42`.  
2. Missing values:  
   - `GarageCars` → 0 (no garage).  
   - Numeric features → median (fit on train only).  
   - Categorical features (`KitchenQual`, `Neighborhood`) → mode.  
3. Encoding:  
   - Numeric → `StandardScaler`.  
   - `KitchenQual` → `OrdinalEncoder` (order: `Po`, `Fa`, `TA`, `Gd`, `Ex`).  
   - `Neighborhood` → `OneHotEncoder`.  

**Models** compared: Linear Regression vs Random Forest (100 trees).  
**Winner**: Random Forest (validation R² = 0.847, test R² = 0.834).  

---

## 🔮 Prompt Versioning

Two prompt variants for Stage 1 were tested on three queries:

| Query | v1 output | v2 output |
|-------|-----------|-----------|
| *3 bed 2 bath colonial with 2 car garage* | `BedroomAbvGr:3, GarageCars:2` | same |
| *Large lot, excellent kitchen, near downtown* | `KitchenQual:Ex` | `KitchenQual:excellent` (invalid) |
| *Fixer-upper with 1 bath and no garage* | `FullBath:1, GarageCars:0` | `FullBath:1, HalfBath:0, GarageCars:0` |

**Winner**: `v1` – because it always returns valid categorical codes (`Ex`, `Gd`, etc.), while `v2` sometimes produces free‑text like `excellent`.

---

## 🚀 Setup & Run Locally

### Prerequisites
- Python 3.11+ (virtual environment recommended)
- Groq API key (free tier) – sign up at [console.groq.com](https://console.groq.com)

### 1. Clone the repository
```bash
git clone https://github.com/mohammadabouamoun/ai-real-estate-agent.git
cd ai-real-estate-agent
2. Create and activate a virtual environment
bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# or .venv\Scripts\activate  # Windows
3. Install dependencies
bash
pip install -r requirements.txt
4. Set up environment variables
Create a .env file (from .env.example) and add your Groq API key:

text
GROQ_API_KEY=your_key_here
5. Run the data cleaning and model training
bash
python src/data_cleaning.py   # splits, imputes, saves processed data
python src/train_model.py     # trains models, saves best_model.pkl
6. Start the FastAPI server
bash
uvicorn src.api:app --reload --port 8000
7. Launch the Streamlit UI (in another terminal)
bash
streamlit run src/ui.py
Then open http://localhost:8501 in your browser.

🐳 Docker
Build the image:

bash
docker build -t ai-real-estate-agent .
Run the container (pass the .env file):

bash
docker run -p 8000:8000 --env-file .env ai-real-estate-agent
The API will be available at http://localhost:8000. You can test it with:

bash
curl -X POST "http://localhost:8000/extract?query=3%20bedroom%20ranch&prompt_version=v1"
The Streamlit UI can be run separately (outside Docker) and will connect to the containerised API.

📓 Colab Notebook
The full EDA, ML pipeline, and prompt versioning experiment are also available as a Google Colab notebook.
Link: AI Real Estate Agent – Colab Notebook
(Replace with your actual shareable link – make sure it’s set to “Anyone with the link → Viewer”)

📈 Results
Validation (Random Forest): RMSE ≈ 37,460 | R² ≈ 0.819

Test (Random Forest): RMSE ≈ 33,982 | R² ≈ 0.834

These scores are from the full 2930‑row dataset. The model explains ~83% of the variance in house prices.

🧪 Error Handling & Gap‑Filling
The /extract endpoint returns a list of missing_fields.

The UI displays input fields for those missing features.

The /predict endpoint rejects requests with missing fields (returns 400).

Stage 2 interpretation includes a fallback message if the LLM call fails.

