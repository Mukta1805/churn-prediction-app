# AI Churn Prediction Tool

A multi-agent system for customer churn prediction using LangGraph, d6tflow, SHAP, and GPT-4.1.

Built as a Columbia University capstone project for Oryx Intelligence.

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Mukta1805/churn-prediction-app.git
cd churn-prediction-app
```

### 2. Create a virtual environment

Conda:

```bash
conda create -n churn-pred python=3.9 -y && conda activate churn-pred
```

venv:

```bash
python -m venv .venv && source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install OpenMP runtime (required)

Some ML libraries require an OpenMP runtime. If you see errors mentioning missing OpenMP libraries, install the appropriate runtime:

- **Windows**: `vcomp140.dll` or `libgomp-1.dll`
- **Mac OSX**: `libomp.dylib`
- **Linux / other UNIX-like OSes**: `libgomp.so`

Mac OSX users: Run `brew install libomp` to install OpenMP runtime.

```bash
brew install libomp
```

### 5. Set up your OpenAI API key

Recommended:

```bash
export OPENAI_API_KEY="your_api_key"
```

Alternative (local dev): copy the example env file and add your key:

```bash
cp .env.example .env
```

You can get an API key at https://platform.openai.com/api-keys

### 6. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## How to Use

1. Upload the included dataset (`data/customer_churn_business_dataset.csv`) or your own CSV with a `churn` column
2. Provide a **Project / Dataset Overview** (text or upload a `.txt`/`.md` file) describing the business context
3. Click **Run Analysis** -- the pipeline will train 4 ML models, compute SHAP explanations, and generate AI insights
4. Explore the three tabs:
   - **Model Results** -- comparison table, ROC-AUC chart, profit curve, SHAP feature importance
     - Includes an **AI explainer + Q&A** to interpret metrics and charts in your project context
   - **Insights** -- AI-generated business insights (churn drivers, at-risk segments, retention strategies)
   - **Chat** -- ask follow-up questions about the analysis

## Architecture

```
LangGraph StateGraph:
  horizon_definition -> class_imbalance -> missing_values -> clean_data
  -> run_model_pipeline (d6tflow) -> compute_shap (SHAP) -> generate_insights (GPT-4.1)
```

- **Agent 1 (Model Selection):** Cleans data, trains Logistic Regression / Gradient Boosting / XGBoost / LightGBM with hyperparameter tuning, selects best model by ROC-AUC, computes SHAP explanations
- **Agent 2 (Insight Generation):** Sends model results to GPT-4.1 to produce business insights and powers interactive Q&A chat
- **Agent 3 (Results Explainer):** Interprets model comparison metrics and charts in plain English, with interactive Q&A (uses the user-provided Project/Dataset Overview for context)

## Tech Stack

- **Orchestration:** LangGraph (stateful agent graph)
- **ML Pipeline:** d6tflow (task caching), scikit-learn, XGBoost, LightGBM
- **Explainability:** SHAP (TreeExplainer / KernelExplainer)
- **AI Insights:** OpenAI GPT-4.1
- **Frontend:** Streamlit
