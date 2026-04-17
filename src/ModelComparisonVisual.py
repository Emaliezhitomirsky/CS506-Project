"""
Model Comparison — Interactive Visualizations

Loads prediction CSVs from all supervised models and generates:
- figures/model_comparison_interactive.html     (actual vs predicted, all models)
- figures/residuals_interactive.html            (residuals, all models)
- figures/model_comparison_bar_interactive.html  (RMSE + R² bar chart)
- results/model_comparison.csv                  (metrics table)

Run AFTER all model scripts:
    python RunLinearRegression.py
    python RunRidgeRegression.py
    python RunRandomForest.py
    python RunXGBoost.py
    python RunSVR.py
    python RunStackingModels.py
    python ModelComparisonVisual.py
"""


import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paths 
OUTPUT_PATH = "results/"
FIG_PATH = "figures/"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# Load predictions from all models 
model_files = {
    "Linear Regression": "linear_regression_predictions.csv",
    "Ridge Regression": "ridge_regression_predictions.csv",
    "Random Forest": "random_forest_predictions.csv",
    "XGBoost": "xgboost_predictions.csv",
    "SVD + SVR": "svd_svr_predictions.csv",
    "Stacking (RF + XGB)": "stacking_predictions.csv",
}

all_predictions = []
model_metrics = []

for model_name, filename in model_files.items():
    filepath = os.path.join(OUTPUT_PATH, filename)
    if os.path.exists(filepath):
        preds = pd.read_csv(filepath)
        preds["model"] = model_name
        preds["residuals"] = preds["actual_energy_per_capita"] - preds["predicted_energy_per_capita"]
        all_predictions.append(preds)

        r2_val = r2_score(preds["actual_energy_per_capita"], preds["predicted_energy_per_capita"])
        rmse_val = np.sqrt(mean_squared_error(preds["actual_energy_per_capita"], preds["predicted_energy_per_capita"]))
        model_metrics.append({"model": model_name, "rmse": rmse_val, "r2": r2_val})
        print(f"Loaded {filename}")
    else:
        print(f"Skipping {model_name} — {filepath} not found")

if not all_predictions:
    print("\nNo prediction files found. Run model scripts first.")
    print("Expected files in results/:")
    for f in model_files.values():
        print(f"  - {f}")
    exit(1)

all_df = pd.concat(all_predictions, ignore_index=True)
metrics_compare = pd.DataFrame(model_metrics)
metrics_compare.to_csv(os.path.join(OUTPUT_PATH, "model_comparison.csv"), index=False)
print("\n")
print("MODEL COMPARISON")
print("\n")
print(metrics_compare.to_string(index=False))

n_models = len(model_metrics)

# Interactive: Actual vs Predicted (all models) 
fig1 = px.scatter(
    all_df,
    x="actual_energy_per_capita",
    y="predicted_energy_per_capita",
    color="model",
    hover_name="country",
    hover_data={
        "year": True,
        "actual_energy_per_capita": ":,.0f",
        "predicted_energy_per_capita": ":,.0f",
        "model": False,
    },
    facet_col="model",
    facet_col_wrap=2,
    title="Actual vs Predicted Energy per Capita — All Models (Hover for Details)",
    labels={
        "actual_energy_per_capita": "Actual (kWh/person)",
        "predicted_energy_per_capita": "Predicted (kWh/person)",
        "model": "Model",
    },
    opacity=0.5,
)

# Add 45-degree reference line to each subplot
max_val = all_df[["actual_energy_per_capita", "predicted_energy_per_capita"]].max().max()
for i in range(1, n_models + 1):
    fig1.add_trace(
        go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines",
            line=dict(color="black", dash="dash", width=1.5),
            showlegend=False,
        ),
        row=(i - 1) // 2 + 1,
        col=(i - 1) % 2 + 1,
    )

# Add R² annotations
for i, row in metrics_compare.iterrows():
    fig1.add_annotation(
        text=f"R²={row['r2']:.3f}<br>RMSE={row['rmse']:,.0f}",
        xref=f"x{i+1 if i > 0 else ''}", yref=f"y{i+1 if i > 0 else ''}",
        x=0.05, y=0.95,
        xanchor="left", yanchor="top",
        showarrow=False,
        font=dict(size=12, color="black"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
    )

fig1.update_layout(
    width=1200, height=900,
    template="plotly_white",
    font=dict(size=11),
    showlegend=False,
)
fig1.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]) if "model=" in a.text else None)

fig1.write_html(os.path.join(FIG_PATH, "model_comparison_interactive.html"))
print(f"\nSaved: {FIG_PATH}model_comparison_interactive.html")

# Interactive: Residuals (all models) 
fig2 = px.scatter(
    all_df,
    x="predicted_energy_per_capita",
    y="residuals",
    color="model",
    hover_name="country",
    hover_data={
        "year": True,
        "actual_energy_per_capita": ":,.0f",
        "predicted_energy_per_capita": ":,.0f",
        "residuals": ":,.0f",
        "model": False,
    },
    facet_col="model",
    facet_col_wrap=2,
    title="Residuals vs Predicted — All Models (Hover to Identify Outliers)",
    labels={
        "predicted_energy_per_capita": "Predicted (kWh/person)",
        "residuals": "Residual (Actual − Predicted)",
        "model": "Model",
    },
    opacity=0.5,
)

for i in range(1, n_models + 1):
    fig2.add_hline(
        y=0, line_dash="dash", line_color="black", line_width=1.5,
        row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1,
    )

fig2.update_layout(
    width=1200, height=900,
    template="plotly_white",
    font=dict(size=11),
    showlegend=False,
)
fig2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]) if "model=" in a.text else None)

fig2.write_html(os.path.join(FIG_PATH, "residuals_interactive.html"))
print(f"Saved: {FIG_PATH}residuals_interactive.html")

# Interactive: Model Comparison Bar Chart 
fig3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=["RMSE (lower is better)", "R² (higher is better)"],
)

colors = px.colors.qualitative.Set2[:n_models]

fig3.add_trace(
    go.Bar(
        x=metrics_compare["model"],
        y=metrics_compare["rmse"],
        marker_color=colors,
        text=metrics_compare["rmse"].apply(lambda x: f"{x:,.0f}"),
        textposition="outside",
        showlegend=False,
    ),
    row=1, col=1,
)

fig3.add_trace(
    go.Bar(
        x=metrics_compare["model"],
        y=metrics_compare["r2"],
        marker_color=colors,
        text=metrics_compare["r2"].apply(lambda x: f"{x:.3f}"),
        textposition="outside",
        showlegend=False,
    ),
    row=1, col=2,
)

fig3.update_layout(
    title="Model Performance Comparison",
    width=1000, height=500,
    template="plotly_white",
    font=dict(size=12),
)
fig3.update_yaxes(title_text="RMSE", row=1, col=1)
fig3.update_yaxes(title_text="R²", row=1, col=2)

fig3.write_html(os.path.join(FIG_PATH, "model_comparison_bar_interactive.html"))
print(f"Saved: {FIG_PATH}model_comparison_bar_interactive.html")

print(f"\nAll outputs saved to {OUTPUT_PATH} and {FIG_PATH}")