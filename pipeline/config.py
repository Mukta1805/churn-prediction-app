"""Constants for the churn prediction pipeline."""

# Business value constants for threshold optimization
BUSINESS_CONSTANTS = {
    "customer_value": 500,
    "contact_cost": 10,
    "retention_success_rate": 0.25,
    "missed_churn_loss": 500,
}

# Models to train (dropped RF — slowest and most memory-hungry)
MODEL_TYPES = ["logreg", "gb", "xgb", "lgbm"]

MODEL_DISPLAY_NAMES = {
    "logreg": "Logistic Regression",
    "gb": "Gradient Boosting",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
}

# Hyperparameter search settings
HYPEROPT_MAX_EVALS = 4

# Columns to drop before modelling
COLUMNS_TO_DROP = ["customer_id", "city"]

# Target column
TARGET_COLUMN = "churn"

# SHAP settings
SHAP_SAMPLE_SIZE = 100
