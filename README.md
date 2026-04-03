# AI Churn Prediction Tool

A multi-agent system for customer churn prediction using LangGraph, d6tflow, SHAP, and GPT-4.1.

Built as a Columbia University capstone project for Oryx Intelligence.

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Mukta1805/churn-prediction-app.git
cd churn-prediction-app
```

### 2. Install dependencies

Requires **Python 3.9+**.

```bash
pip install -r requirements.txt
```

### 3. Set up your OpenAI API key

Copy the example env file and add your key:

```bash
cp .env.example .env
```

Then edit `.env` and replace the placeholder with your actual OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
```

You can get an API key at https://platform.openai.com/api-keys

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## How to Use

1. Upload the included dataset (`data/customer_churn_business_dataset.csv`) or your own CSV with a `churn` column
2. Click **Run Analysis** -- the pipeline will train 4 ML models, compute SHAP explanations, and generate AI insights
3. Explore the three tabs:
   - **Model Results** -- comparison table, ROC-AUC chart, profit curve, SHAP feature importance
   - **Insights** -- AI-generated business insights (churn drivers, at-risk segments, retention strategies)
   - **Chat** -- ask follow-up questions about the analysis

## Architecture

```
LangGraph StateGraph:
  clean_data -> run_model_pipeline (d6tflow) -> compute_shap (SHAP) -> generate_insights (GPT-4.1)
```

- **Agent 1 (Model Selection):** Cleans data, trains Logistic Regression / Gradient Boosting / XGBoost / LightGBM with hyperparameter tuning, selects best model by ROC-AUC, computes SHAP explanations
- **Agent 2 (Insight Generation):** Sends model results to GPT-4.1 to produce business insights and powers interactive Q&A chat

## Tech Stack

- **Orchestration:** LangGraph (stateful agent graph)
- **ML Pipeline:** d6tflow (task caching), scikit-learn, XGBoost, LightGBM
- **Explainability:** SHAP (TreeExplainer / KernelExplainer)
- **AI Insights:** OpenAI GPT-4.1
- **Frontend:** Streamlit
