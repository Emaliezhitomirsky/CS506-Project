import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVR
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

DATA_PATH="Data/owid-energy-data-clean.csv"
OUTPUT_PATH="results/"
FIG_PATH="figures/"
MODEL_NAME="svd_svr"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

df=pd.read_csv(DATA_PATH)
df=df.sort_values(["country", "year"])

train=df[df["year"] <= 2012]
test=df[df["year"] > 2012]

feature_cols=[
    "year", "log_population", "log_gdp_per_capita",
    "coal_share_energy", "gas_share_energy", "oil_share_energy",
    "biofuel_share_energy",
    "hydro_share_energy", "solar_share_energy", "wind_share_energy",
    "nuclear_share_energy",
]

X_train_raw=train[feature_cols]
X_test_raw=test[feature_cols]
y_train=np.log1p(train["energy_per_capita"])
y_test=np.log1p(test["energy_per_capita"])

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train_raw)
X_test_scaled=scaler.transform(X_test_raw)

svd=TruncatedSVD(n_components=5, random_state=42)
X_train=svd.fit_transform(X_train_scaled)
X_test=svd.transform(X_test_scaled)

model=SVR(kernel='rbf', C=1, epsilon=0.1, gamma='scale')

model.fit(X_train, y_train)

y_pred=model.predict(X_test)
y_pred_actual=np.expm1(y_pred)
y_test_actual=np.expm1(y_test)

results_df=pd.DataFrame({
    "year": test["year"],
    "country": test["country"],
    "actual_energy_per_capita": y_test_actual,
    "predicted_energy_per_capita": y_pred_actual
})

results_file=os.path.join(OUTPUT_PATH, f"{MODEL_NAME}_predictions.csv")
results_df.to_csv(results_file, index=False)

rmse=np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
r2=r2_score(y_test_actual, y_pred_actual)

metrics_df=pd.DataFrame({
    "model": ["SVD + SVR"],
    "rmse": [rmse],
    "r2": [r2]
})

metrics_file=os.path.join(OUTPUT_PATH, f"{MODEL_NAME}_metrics.csv")
metrics_df.to_csv(metrics_file, index=False)

print(f"Saved predictions to {results_file}")
print(metrics_df)

plt.figure(figsize=(10, 6))
plt.scatter(
    results_df["actual_energy_per_capita"],
    results_df["predicted_energy_per_capita"],
    alpha=0.4, s=40, color='teal'
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
plt.title("Actual vs Predicted Energy per Capita (SVD + SVR)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, f"{MODEL_NAME}_actual_vs_pred.png"))
plt.close()

residuals=results_df["actual_energy_per_capita"] - results_df["predicted_energy_per_capita"]
results_df["residuals"]=residuals

plt.figure(figsize=(10, 6))
plt.scatter(
    results_df["predicted_energy_per_capita"],
    results_df["residuals"],
    alpha=0.4, s=40, color='crimson'
)
plt.axhline(0, color='black', linestyle='--', linewidth=2)
plt.xlabel("Predicted Energy per Capita")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted (SVD + SVR)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, f"{MODEL_NAME}_residuals.png"))
plt.close()

print(f"Saved figures to {FIG_PATH}")