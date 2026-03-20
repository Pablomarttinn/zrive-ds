import json
import os
import logging
from datetime import datetime
 
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
 

TARGET_COL = "outcome"
 
# Features used for training — drop IDs, dates, and the target
DROP_COLS = [
    "variant_id",
    "order_id",
    "user_id",
    "created_at",
    "order_date",
    TARGET_COL,
]
 
TRAINING_COLS = [
    "user_order_seq",
    "ordered_before",
    "abandoned_before",
    "active_snoozed",
    "set_as_regular",
    "normalised_price",
    "discount_pct",
    "global_popularity",
    "count_adults",
    "count_children",
    "count_babies",
    "count_pets",
    "people_ex_baby",
    "days_since_purchase_variant_id",
    "avg_days_to_buy_variant_id",
    "std_days_to_buy_variant_id",
    "days_since_purchase_product_type",
    "avg_days_to_buy_product_type",
    "std_days_to_buy_product_type",
]

MODEL_DIR = "/home/pablomartin/ZriveDs/zrive-ds/models"
DATA_PATH = "/home/pablomartin/ZriveDs/zrive-ds/data/module3/feature_frame.csv"
 
def load_data_csv(path: str) -> pd.DataFrame:
    """Load training data from a local CSV file."""
    logger.info("Loading data from CSV: %s", path)
    return pd.read_csv(path)

def create_wanted_df(df:pd.DataFrame, min_products: int = 5) -> pd.DataFrame:
    """we want only orders with more than five products"""
    order_size = df.groupby("order_id")["outcome"].sum()
    orders_wanted = order_size[order_size >= min_products].index
    return df.loc[lambda x: x.order_id.isin(orders_wanted)]



def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    logger.info("Preprocessing data — shape before: %s", df.shape)
 
    df_selected = df.pipe(create_wanted_df)
    
 
    X = df_selected[TRAINING_COLS]
    y = df_selected[TARGET_COL]
 
    logger.info("Preprocessing done — X shape: %s | positive-rate: %.3f", X.shape, y.mean())
    return X, y

def build_pipeline(params: dict) -> Pipeline:

    rf_params = {
        "n_estimators":      params.get("n_estimators",      100),
        "max_depth":         params.get("max_depth",          None),
        "min_samples_split": params.get("min_samples_split",  2),
        "min_samples_leaf":  params.get("min_samples_leaf",   1),
        "max_features":      params.get("max_features",       "sqrt"),
        "random_state":      params.get("random_state",       42),
        "n_jobs":            params.get("n_jobs",             -1),
    }
    logger.info("RandomForest params: %s", rf_params)
 
    pipeline = Pipeline([
        ("clf",    RandomForestClassifier(**rf_params)),
    ])
    return pipeline

def save_model(pipeline: Pipeline, model_dir: str, model_name: str) -> str:

    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    joblib.dump(pipeline, model_path)
    logger.info("Model saved → %s", model_path)
    return os.path.abspath(model_path)

def handler_fit(event: dict, _) -> dict:

    try:
 
        df = load_data_csv( path=DATA_PATH)

        X, y = preprocess(df)
        params = event['model_parametrisation']
        pipeline = build_pipeline(params)
        logger.info("Fitting model on %d samples …", len(X))
        pipeline.fit(X, y)
        logger.info("Training complete.")
 
        
        today = datetime.today().strftime("%Y_%m_%d")
        model_name = f"push_{today}"
        model_path = save_model(pipeline, MODEL_DIR, model_name)
 
        return {
            "statusCode": "200",
            "body": json.dumps({
                "model_name": model_name,
                "model_path": model_path,
            }),
        }
 
    except Exception as exc:          
        logger.exception("handler_fit failed: %s", exc)
        return {
            "statusCode": "500",
            "body": json.dumps({"error": str(exc)}),
        }
 


if __name__ == "__main__":
    sample_event = {
        "model_parametrisation": {
            "n_estimators": 200,
            "max_depth":    10,
            "random_state": 0,
        }
    }
    result = handler_fit(sample_event, None)
    print(result)
 