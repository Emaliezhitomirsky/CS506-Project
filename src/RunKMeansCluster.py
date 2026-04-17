"""
K-Means Clustering + PCA on Energy Dataset
- Use SVD/PCA to identify components capturing most variance across countries
- Use K-Means to segment countries into distinct groups by wealth, energy profile,
  and consumption patterns

Features used:
- Energy Consumption per Capita
- GDP per Capita
- Population (log)
- Fossil Fuel % and Renewable Energy %

Input:
- Data/owid-energy-data-clean.csv

Output:
- figures/elbow_silhouette_plot.png
- figures/pca_variance_explained.png
- figures/kmeans_clusters_interactive.html  (interactive Plotly scatter)
- results/cluster_assignments.csv
- results/pca_components.csv
- results/clustering_metrics.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# Paths 
DATA_PATH = "Data/owid-energy-data-clean.csv"
OUTPUT_PATH = "results/"
FIG_PATH = "figures/"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# Load 
df = pd.read_csv(DATA_PATH)

# Use most recent year snapshot so each country is one row
SNAPSHOT_YEAR = df["year"].max()
snapshot = df[df["year"] == SNAPSHOT_YEAR].copy()

# Combine energy types
snapshot["fossil_share_energy"] = (
    snapshot["coal_share_energy"]
    + snapshot["gas_share_energy"]
    + snapshot["oil_share_energy"]
)
snapshot["renewables_share_energy"] = (
    snapshot["hydro_share_energy"]
    + snapshot["solar_share_energy"]
    + snapshot["wind_share_energy"]
    + snapshot["biofuel_share_energy"]
)

# Clustering features
cluster_features = [
    "energy_per_capita",
    "log_gdp_per_capita",
    "log_population",
    "fossil_share_energy",
    "renewables_share_energy",
]

snapshot = snapshot.dropna(subset=cluster_features)
X_raw = snapshot[cluster_features].values
countries = snapshot["country"].values

print(f"Clustering {len(countries)} countries using {SNAPSHOT_YEAR} snapshot")
print(f"Features: {cluster_features}\n")

# Standardize 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# PCA for diamensions reduction
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# PC1 - Wealth + clean energy. High GDP, high renewables, low fossil = high PC1 score.
# PC2 - Energy intensity. Massive fossil fuel consumption = high PC2 score
# PC3 - Population size. Large population = high PC3 score
# PC4 - Wealth vs. energy efficiency. High GDP but low energy per capita = high PC4 score
# PC5 - General Noise?

# Variance explained
cumulative_var = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(cumulative_var >= 0.90) + 1

print("PCA Explained Variance Ratios:")
for i, (var, cum) in enumerate(zip(pca.explained_variance_ratio_, cumulative_var)):
    print(f"  PC{i+1}: {var:.3f}  (cumulative: {cum:.3f})")
print(f"\nComponents needed for 90% variance: {n_components_90}\n")

# Save PCA loadings
loadings_df = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(cluster_features))],
    index=cluster_features,
)
loadings_df.to_csv(os.path.join(OUTPUT_PATH, "pca_components.csv"))


# Plot variance explained
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(1, len(pca.explained_variance_ratio_) + 1),
       pca.explained_variance_ratio_, alpha=0.6, label="Individual")
ax.step(range(1, len(cumulative_var) + 1), cumulative_var,
        where="mid", color="red", label="Cumulative")
ax.axhline(0.90, color="gray", linestyle="--", linewidth=1, label="90% threshold")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance Ratio")
ax.set_title("PCA Variance Explained")
ax.set_xticks(range(1, len(cluster_features) + 1))
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, "pca_variance_explained.png"), dpi=150)
plt.close()

# OPTIMIZING 

# Find Elbow + Silhouette scores
K_RANGE = range(2, 11)
inertias = []
silhouettes = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# best k
# but k=9 give a 2 countries clusters (mainly outliers) with only a 0.004 from k=4
best_k = list(K_RANGE)[np.argmax(silhouettes)]
print(f"Best k by silhouette score: {best_k}")

CHOSEN_K = 4
km_final = KMeans(n_clusters=CHOSEN_K, n_init=30, random_state=42) #30 times
cluster_labels = km_final.fit_predict(X_scaled)

snapshot = snapshot.copy()
snapshot["cluster"] = cluster_labels

# Save 
assignments = snapshot[["country", "iso_code", "cluster"] + cluster_features].copy()
assignments.to_csv(os.path.join(OUTPUT_PATH, "kmean_cluster_assignments.csv"), index=False)

metrics_df = pd.DataFrame({
    "model": ["KMeans"],
    "k": [CHOSEN_K],
    "silhouette_score": [silhouette_score(X_scaled, cluster_labels)],
    "inertia": [km_final.inertia_],
})
metrics_df.to_csv(os.path.join(OUTPUT_PATH, "kmean_clustering_metrics.csv"), index=False)
print(metrics_df)

# Cluster summaries
print(f"\nKMeans Clustering (k={CHOSEN_K}, snapshot year={SNAPSHOT_YEAR})")
for c in range(CHOSEN_K):
    members = snapshot[snapshot["cluster"] == c]
    print(f"\nCluster {c} ({len(members)} countries):")
    print(f"  Countries: {', '.join(sorted(members['country'].values))}")
    print(f"  Avg Energy/Capita:   {members['energy_per_capita'].mean():>10,.0f} kWh")
    print(f"  Avg GDP/Capita:      ${np.expm1(members['log_gdp_per_capita']).mean():>10,.0f}")
    print(f"  Avg Fossil %:        {members['fossil_share_energy'].mean():>10.1f}%")
    print(f"  Avg Renewables %:    {members['renewables_share_energy'].mean():>10.1f}%")

#Result Plots

# Elbow + Silhouette plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(list(K_RANGE), inertias, "o-", linewidth=2)
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
ax1.set_title("Elbow Method for Optimal k")
ax1.set_xticks(list(K_RANGE))
ax2.plot(list(K_RANGE), silhouettes, "o-", linewidth=2, color="green")
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score vs Number of Clusters")
ax2.set_xticks(list(K_RANGE))

plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, "elbow_silhouette_plot.png"), dpi=150)
plt.close()

cluster_names = {
    0: "Industrialized",
    1: "Developing",
    2: "Renewable Leaders",
    3: "Petrostates",
}
cluster_colors = {
    0: "#66c2a5",
    1: "#8da0cb",
    2: "#ffd92f",
    3: "#fc4235",
}

# Interactive Visualization - PCA Scatter 
plot_df = snapshot[["country", "cluster"] + cluster_features].copy()
plot_df["PC1"] = X_pca[:, 0]
plot_df["PC2"] = X_pca[:, 1]
plot_df["cluster_label"] = plot_df["cluster"].map(cluster_names)
# Reverse log transforms for readable hover
plot_df["gdp_per_capita"] = np.expm1(plot_df["log_gdp_per_capita"]).round(0)
plot_df["population"] = np.expm1(plot_df["log_population"]).round(0)
 
fig1 = px.scatter(
    plot_df,
    x="PC1", y="PC2",
    color="cluster_label",
    hover_name="country",
    hover_data={
        "gdp_per_capita": ":,.0f",
        "energy_per_capita": ":,.0f",
        "fossil_share_energy": ":.1f",
        "renewables_share_energy": ":.1f",
        "PC1": False,
        "PC2": False,
        "cluster_label": False,
    },
    title=f"Country Clusters in PCA Space ({SNAPSHOT_YEAR}) — Hover for Details",
    labels={
        "PC1": f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        "PC2": f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        "cluster_label": "Cluster",
        "gdp_per_capita": "GDP/Capita ($)",
        "energy_per_capita": "Energy/Capita (kWh)",
        "fossil_share_energy": "Fossil %",
        "renewables_share_energy": "Renewables %",
    },
    color_discrete_map={name: cluster_colors[i] for i, name in cluster_names.items()},
    text="country",
)
 
fig1.update_traces(
    textposition="top center",
    textfont_size=8,
    marker=dict(size=12, line=dict(width=1, color="DarkSlateGrey")),
)
fig1.update_layout(
    width=1100, height=700,
    template="plotly_white",
    font=dict(size=12),
)
 
fig1.write_html(os.path.join(FIG_PATH, "kmeans_clusters_interactive.html"))
print(f"\nSaved interactive plot: {FIG_PATH}kmeans_clusters_interactive.html")