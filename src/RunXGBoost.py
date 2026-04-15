"""
Train, evaluate, and plot XGBoost on energy dataset

- Time-based train/test split (1965–2012 train, 2013–2024 test)
- Trains XGBoost
- Evaluates RMSE and R²
- Plots Actual vs Predicted and Residuals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Paths and config
DATA_PATH = "Data/owid-energy-data-clean.csv"
OUTPUT_PATH = "results/"
FIG_PATH = "figures/"
MODEL_NAME = "xgboost"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
df = df.sort_values(["country", "year"])

# Time-based train/test split
train = df[df["year"] <= 2012]
test  = df[df["year"] > 2012]

feature_cols = [
    "year", "log_population", "log_gdp_per_capita",
    "coal_share_energy", "gas_share_energy", "oil_share_energy",
    "biofuel_share_energy",
    "hydro_share_energy", "solar_share_energy", "wind_share_energy",
    "nuclear_share_energy",
]

X_train = train[feature_cols]
X_test  = test[feature_cols]

y_train = np.log1p(train["energy_per_capita"])
y_test  = np.log1p(test["energy_per_capita"])

# Train XGBoost
model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_pred_actual = np.expm1(y_pred)
y_test_actual = np.expm1(y_test)

results_df = pd.DataFrame({
    "year": test["year"],
    "country": test["country"],
    "actual_energy_per_capita": y_test_actual,
    "predicted_energy_per_capita": y_pred_actual
})

results_file = os.path.join(OUTPUT_PATH, f"{MODEL_NAME}_predictions.csv")
results_df.to_csv(results_file, index=False)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
r2   = r2_score(y_test_actual, y_pred_actual)

metrics_df = pd.DataFrame({
    "model": ["XGBoost"],
    "rmse": [rmse],
    "r2": [r2]
})

metrics_file = os.path.join(OUTPUT_PATH, f"{MODEL_NAME}_metrics.csv")
metrics_df.to_csv(metrics_file, index=False)

print(f"Saved predictions to {results_file}")
print(metrics_df)

# Actual vs Predicted Plot
plt.figure(figsize=(10, 6))
plt.scatter(
    results_df["actual_energy_per_capita"],
    results_df["predicted_energy_per_capita"],
    alpha=0.4,
    s=40,
    color='steelblue'
)
sns.regplot(
    x="actual_energy_per_capita",
    y="predicted_energy_per_capita",
    data=results_df,
    scatter=False,
    color="black",
    line_kws={"linewidth": 2, "linestyle": "--"}
)
plt.xlabel("Actual Energy per Capita")
plt.ylabel("Predicted Energy per Capita")
plt.title("Actual vs Predicted Energy per Capita (XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, f"{MODEL_NAME}_actual_vs_pred.png"))
plt.close()

# Residual Plot
residuals = results_df["actual_energy_per_capita"] - results_df["predicted_energy_per_capita"]
results_df["residuals"] = residuals

plt.figure(figsize=(10, 6))
plt.scatter(
    results_df["predicted_energy_per_capita"],
    results_df["residuals"],
    alpha=0.4,
    s=40,
    color='tomato'
)
plt.axhline(0, color='black', linestyle='--', linewidth=2)
plt.xlabel("Predicted Energy per Capita")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted (XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, f"{MODEL_NAME}_residuals.png"))
plt.close()

print(f"Saved figures to {FIG_PATH}")