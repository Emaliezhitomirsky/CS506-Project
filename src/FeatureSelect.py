"""
Feature selection and basic preprocessing
Goal:
- Select relevant features for predicting energy_per_capita.
- Filter dataset by year and countries with sufficient data.
- Drop rows with missing target.
"""
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Data/owid-energy-data-updated.csv")

# Feature selection
selected_columns = [
    "country", "year", "iso_code", "population", "gdp",
    "coal_share_energy", "gas_share_energy", "oil_share_energy",
    "biofuel_share_energy",
    "hydro_share_energy", "solar_share_energy", "wind_share_energy",
    "nuclear_share_energy",
    "energy_per_capita"  # target
]
df_selected = df[selected_columns].copy()
df_selected['gdp_per_capita'] = df_selected['gdp'] / df_selected['population']
df_selected.drop(columns=['gdp'], inplace=True)

# These columns aren't linearly distributed, so we also add log-transformed versions.
# Both raw and log versions are kept for flexibility in modelling.
skewed_cols = ["gdp_per_capita", "population"]
for col in skewed_cols:
    df_selected[f"log_{col}"] = np.log1p(df_selected[col])
# raw gdp_per_capita and population are retained as-is

# Filter by Year and Country
df_selected = df_selected[df_selected["year"] >= 1965]

# Drop aggregates. Non-aggregate entries have 3-letter ISO codes
df_selected = df_selected[df_selected['iso_code'].str.len() == 3]

# Drop countries with >40% missing
threshold = 0.4
country_missing = df_selected.groupby("country").apply(lambda x: x.isna().mean().mean())
good_countries = country_missing[country_missing <= threshold].index
df_selected = df_selected[df_selected["country"].isin(good_countries)]

# Drop rows with missing target
df_selected = df_selected.dropna(subset=['energy_per_capita']).copy()

# Save cleaned dataset
output_file = "Data/owid-energy-data-selected.csv"
df_selected.to_csv(output_file, index=False)
print(f"Cleaned dataset saved: {output_file}")
print(f"Shape after filtering: {df_selected.shape}")
print(f"Remaining countries: {df_selected['country'].nunique()}")
print(f"Missing values:\n{df_selected.isna().mean().sort_values(ascending=False)}")