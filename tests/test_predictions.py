import pandas as pd
import numpy as np

def test_xgboost_predictions_valid():
    df = pd.read_csv("results/xgboost_predictions.csv")

    assert "predicted_energy_per_capita" in df.columns
    assert "actual_energy_per_capita" in df.columns

    preds = df["predicted_energy_per_capita"]

    assert not preds.isna().any()
    assert np.isfinite(preds).all()

    assert (preds >= 0).all()

    assert preds.std() > 0