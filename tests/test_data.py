import pandas as pd

def test_data_load():
    df = pd.read_csv("Data/owid-energy-data-clean.csv")
    assert df.shape[0] > 1000
    assert "energy_per_capita" in df.columns