import pandas as pd

def test_predictions_exist():
    df = pd.read_csv("results/xgboost_predictions.csv")
    assert "predicted_energy_per_capita" in df.columns
    assert len(df) > 0