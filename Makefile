install:
	pip install -r requirements.txt

run-all:
	python src/OverwriteGDP.py
	python src/FeatureSelect.py
	python src/CleanData.py
	python src/PrelimVisuals.py
	python src/RunLinearRegression.py
	python src/RunRidgeRegression.py
	python src/RunRandomForest.py
	python src/RunXGBoost.py
	python src/RunSVR.py
	python src/RunStackingModels.py
	python src/ModelComparisonVisual.py
	python src/RunKmeansCluster.py