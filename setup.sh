# setup.sh
python -m src.preprocess
python -m src.feature_engineering
python -m src.hyperparameter_grid
python -m src.model_eval
python -m src.forecast
