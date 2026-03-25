import os
import logging
import pandas as pd
import joblib
import argparse

from .train import MODEL_DIR

logger = logging.getLogger(__name__)
logger.level = logging.INFO

consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)

PREDICTIONS_DIR = "/home/pablomartin/ZriveDs/zrive-ds/predicitions"


def save_predictions(y_pred: pd.Series) -> None:

    df = pd.DataFrame({"order_id" : y_pred.index, "buy_prediction": y_pred})
    path = os.path.join(PREDICTIONS_DIR,"predicitons_csv")
    df.to_csv(os.path.join(path, index = False))

    logger.info(f"predictions saved in {path}")


def main(data_path: str):
    model_name = "ridge_model.pkl"
    model_path = os.path.join(MODEL_DIR, model_name)
    model = joblib.load(model_path)

    logger.info(f"loaded model {model_name}")

    X_data = pd.read_csv(data_path)

    y_pred = model.predict_proba(X_data)[:,1]

    save_predictions(y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_data_path", type=str, help="Path to input dataset:") #esperando que sea el path del data set listo para predecir
    args = parser.parse_args()

    main(args.x_data_path)