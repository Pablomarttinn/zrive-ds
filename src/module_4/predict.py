import json 
import logging
import os

import pandas as pd
import joblib
import numpy as np


logging.basicConfig(level = logging.INFO, format = "%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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

_MODEL_CACHE: dict[str, object] = {} # We use the model cache to store already loaded models so that if we re run the file,
# memory is optimised by not re-loading the model into memory
 
 
def load_model(model_path: str):
   
    model_path = os.path.abspath(model_path)
    if model_path not in _MODEL_CACHE:
        logger.info("Loading model from disk: %s", model_path)
        _MODEL_CACHE[model_path] = joblib.load(model_path)
    else:
        logger.info("Using cached model: %s", model_path)
    return _MODEL_CACHE[model_path]

def align_features(df: pd.DataFrame, pipeline) -> pd.DataFrame:
 
    # Attempt to retrieve the exact feature list the model was fitted on
    try:
        expected_cols = list(pipeline.named_steps["clf"].feature_names_in_)
        logger.info("Using %d features from model.feature_names_in_", len(expected_cols))

    except AttributeError:
        #fall back to TRAINING_COLS
        expected_cols = TRAINING_COLS
        logger.warning(
            "Could not read feature_names_in_ from model; "
            "falling back to hard-coded TRAINING_COLS (%d cols).",
            len(expected_cols),
        )
 
    # Fill missing columns with 0
    missing = set(expected_cols) - set(df.columns)
    if missing:
        logger.warning("Filling %d missing feature(s) with 0: %s", len(missing), missing)
        for col in missing:
            df[col] = 0
 
    # Drop extra columns
    extra = set(df.columns) - set(expected_cols)
    if extra:
        logger.debug("Dropping %d unexpected feature(s): %s", len(extra), extra)
        df = df.drop(columns=list(extra))
 
    return df[expected_cols]


def predict(X: pd.DataFrame, pipeline, threshold: float = 0.5, return_class: bool = False):
    

    proba = pipeline.predict_proba(X)[:, 1]  
    if return_class:
        return (proba >= threshold).astype(int) # Adding the return class parameter allows 
    #the user to decide between seen the probabilities or the classes.
    return proba

def handler_predict(event: dict, _) -> dict:

    try:
        
        model_path    = event["model_path"]
        users_json    = event["users"]                        # JSON string
        threshold     = float(event.get("proba_threshold", 0.5))
        return_class  = bool(event.get("return_class", False))
 
        
        raw_dict = json.loads(users_json)
        data_to_predict = pd.DataFrame.from_dict(raw_dict).T  
        data_to_predict.index.name = "user_id"
        logger.info("Received %d user(s) for inference.", len(data_to_predict))
 
        
        data_to_predict = data_to_predict.apply(pd.to_numeric, errors="coerce")
        if data_to_predict.isna().any().any():
            nan_cols = data_to_predict.columns[data_to_predict.isna().any()].tolist()
            logger.warning(f"NaN values detected in columns: {nan_cols}")
 
        
        pipeline = load_model(model_path)
 
        
        X = align_features(data_to_predict, pipeline)
 
        
        preds = predict(X, pipeline, threshold=threshold, return_class=return_class)
 
        
        prediction_dict = {
            str(id): round(float(p), 4)
            for id, p in zip(data_to_predict.index, preds)
        }

        return {
            "statusCode": "200",
            "body": json.dumps({"prediction": prediction_dict}),
        }
 
    except KeyError as ke:
        logger.error("Missing required event key: %s", ke)
        return {
            "statusCode": "400",
            "body": json.dumps({"error": f"Missing required field: {ke}"}),
        }
    
    except Exception as exc:         
        logger.exception("handler_predict failed: %s", exc)
        return {
            "statusCode": "500",
            "body": json.dumps({"error": str(exc)}),
        }
 

if __name__ == "__main__":
    sample_event = {
        "model_path": "models/push_2026_03_20.joblib",
        "proba_threshold": 0.5,
        "return_class": False,
        "users": json.dumps({
            "user_42": {
                "user_order_seq": 3,
                "ordered_before": 1,
                "abandoned_before": 0,
                "active_snoozed": 0,
                "set_as_regular": 1,
                "normalised_price": 0.45,
                "discount_pct": 0.1,
                "global_popularity": 0.78,
                "count_adults": 2,
                "count_children": 1,
                "count_babies": 0,
                "count_pets": 0,
                "people_ex_baby": 3,
                "days_since_purchase_variant_id": 14,
                "avg_days_to_buy_variant_id": 10,
                "std_days_to_buy_variant_id": 3,
                "days_since_purchase_product_type": 7,
                "avg_days_to_buy_product_type": 8,
                "std_days_to_buy_product_type": 2,
            }
        }),
    }
    result = handler_predict(sample_event, None)
    print(result)
