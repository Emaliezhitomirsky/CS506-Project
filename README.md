# CS506 Project — Global Energy Consumption Modeling

**Final Video Presentation:** https://youtube.com/YOUR-LINK-HERE

---

## 1. How to Run the Project 

To reproduce all results, run:

```bash
git clone https://github.com/JiaLiiin/CS506-Project
cd CS506-Project
make install
make run-all
```

### What This Does

The `Makefile` executes the full pipeline:

- Data cleaning and preprocessing  
- Feature engineering  
- Exploratory data analysis (EDA)  
- Model training:
  - Linear Regression  
  - Ridge Regression  
  - Random Forest  
  - XGBoost  
  - SVR (SVD + SVR pipeline)  
  - Stacking Ensemble  
- Model comparison visualizations  
- K-Means clustering analysis  

---

## 2. Installation Requirements

Optional manual install:

```bash
pip install -r requirements.txt
```

### Main Dependencies

- pandas
- requests
- matplotlib
- numpy
- Pillow
- scikit-learn
- seaborn
- xgboost
- plotly 

---

## 3. Tests + GitHub Workflow

### Tests

Located in `/tests`.

Run locally:

```bash
pytest
```

### Tests Include

- Data loading validation  
- Required column checks  
- Prediction file existence checks  
- Prediction sanity checks:
  - no NaNs  
  - non-negative values  

### GitHub Actions (CI)

Workflow file:

.github/workflows/tests.yml

Automatically:

- installs dependencies  
- runs tests on every push / pull request  

### Purpose

- Ensures code runs correctly  
- Validates model outputs  
- Prevents broken commits  

---

## 4. Project Overview

This project analyzes global energy consumption patterns and their relationship with economic indicators at the country level, aiming to study how energy consumption varies across countries and over time.

The primary goal of this project is to predict energy consumption per capita using economic and energy-related features such as GDP, population, and energy mix composition. Additionally, we aim to analyze the relationship between GDP per capita and energy consumption per capita using GDP, population density, and fossil fuel share as features and examine trends in fossil fuel and renewable energy usage across countries over time using statistical analysis and visualizations.

---

## 5. Dataset

This project combines:

- OWID Energy Dataset (Our World in Data)  
- World Bank GDP Data  

### Final Dataset

- Time range: 1965–2024  
- Countries: 69  
- Rows after cleaning: 3,885  

---

## 6. Data Processing

### Data Cleaning
- Merged OWID dataset with World Bank GDP data
- Removed aggregated regions
- Filtered countries with excessive missing data

### Missing Values
- GDP: interpolated within country
- Energy shares: forward/backward fill
- Dropped rows missing target (~15 rows)

### Feature Engineering
- log_gdp_per_capita
- log_population
- energy mix shares (coal, gas, oil, nuclear, hydro, wind, solar, biofuel)

### Target Transformation
- energy_per_capita transformed using log1p
- predictions converted back using expm1

---

## 7. Exploratory Data Analysis (EDA)

### Key Insights

- Energy consumption is highly unequal across countries  
- Developed countries plateau at high consumption levels  
- China shows rapid growth post-2000  
- Energy transitions differ significantly:
  - Germany → strong renewable transition  
  - Brazil → balanced mix  
  - India → slow structural change  

### Key Relationship

- GDP per capita strongly correlates with energy use at low income levels  
- Relationship weakens at higher income levels  

This indicates a strong **nonlinear relationship** between economic development and energy consumption.

---

## 8. Modeling Methods

We use supervised learning to predict energy consumption per capita.

### Models Tested
- Linear Regression
- Ridge Regression
- Random Forest
- XGBoost (best model)
- SVD + SVR
- Stacking Ensemble (RF + XGBoost)

### Features Used
- `log_gdp_per_capita`  
- `log_population`  
- Energy mix shares:
  - coal  
  - gas  
  - oil  
  - nuclear  
  - hydro  
  - wind  
  - solar  
  - biofuel 

These features were selected because they capture the primary drivers of per capita energy consumption. GDP per capita reflects economic activity, infrastructure development, and overall wealth, all of which strongly influence energy use. Population captures the scale and distribution of demand, with log transformations helping account for skewness and non-linear growth patterns. Energy mix shares (expressed as percentages of total consumption) provide insight into how energy is produced and consumed, allowing the model to capture differences in efficiency, technology, and resource dependence across countries.

Aggregate features such as total renewable and fossil energy shares were removed because they are redundant with their component variables. Additionally, greenhouse gas emissions were excluded due to high missingness, making reliable imputation impractical.

---

## 9. Model Comparison System 

We built a centralized evaluation script:

ModelComparisonVisual.py

This generates:
- interactive actual vs predicted plots
- residual plots for all models
- RMSE and R² comparison bar charts
- combined metrics table

---

## 10. Model Results

| Model | RMSE | R² |
|------|------|------|
| Linear Regression | 28,540 | 0.550 |
| Ridge Regression | 27,155 | 0.589 |
| Random Forest | 21,871 | 0.734 |
| XGBoost | **17,338** | **0.833** |
| Stacking | 17,524 | 0.829 |
| SVR (SVD + SVR) | 26,753 | 0.601 |

### Key Findings

- Tree-based models significantly outperform linear models  
- Stacking provides minimal improvement  
- Errors increase for high-energy-consuming countries 
- Overall, XGBoost achieves the best performance, confirming that nonlinear, tree-based methods are well-suited for modeling global energy consumption patterns. 

---

## 11. Clustering Analysis (K-Means + PCA)

We applied K-Means clustering and PCA to identify country-level energy regimes.

### Features Used
- energy per capita  
- GDP per capita  
- population (log)  
- fossil fuel share  
- renewable share 

### Results

- Optimal clusters: 4  
- Silhouette Score: ~0.30 

### Clusters Identified
- Industrialized economies  
- Developing economies  
- Renewable leaders  
- Fossil-fuel-heavy economies (petrostates)  

PCA revealed that GDP and energy structure explain most variance in global energy systems.

---

## 12. Visualizations

Visualizations are stored in `/figures`.

### Interactive (Plotly)

- Model comparison (actual vs predicted)  
- Residual analysis  
- PCA clustering visualization  

### Static

- PCA variance explained  
- Elbow method plot  
- Silhouette plot  

---

## 13. Conclusion

Our results show that we successfully predict energy consumption per capita with high accuracy (R² ≈ 0.83 using XGBoost), achieving the primary objective of the project.

This project demonstrates that global energy consumption patterns are highly nonlinear and cannot be effectively captured using simple linear models. While economic development, measured by GDP per capita, is a strong driver of energy use, its influence varies across different stages of development and interacts with factors such as population and energy mix composition. 

Machine learning approaches, particularly tree-based models like XGBoost, significantly improve predictive performance by capturing these complex relationships. 

Additionally, clustering analysis reveals that countries naturally group into distinct energy consumption regimes, reflecting structural differences in development and energy systems. 

Overall, the results highlight both the predictive power of advanced models and the importance of accounting for heterogeneity across countries when modeling global energy consumption.