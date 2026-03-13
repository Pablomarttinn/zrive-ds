import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import auc, precision_recall_curve,roc_auc_score
from sklearn.linear_model import LogisticRegression

import logging
import datetime
import os
from pathlib import Path    
import joblib

logger = logging.getLogger(__name__)
logger.level = logging.INFO
handler = logging.StreamHandler()
logger.addHandler(handler)

STORAGE_PATH = "/home/pablomartin/ZriveDs/zrive-ds/data/module3/feature_frame.csv"
MODEL_DIR = Path("/home/pablomartin/ZriveDs/zrive-ds/models")

RIDGE_C = 1e-4
FEATURE_COLS = [
    "ordered_before","global_popularity","abandoned_before","set_as_regular","normalised_price"
]

TARGET_COL = "outcome"
TRAIN_PERCENT = 0.8



def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(STORAGE_PATH)
    logger.info(f"Dataset loaded with shape {df.shape}")
    return df

def relevant_dataset(df: pd.DataFrame, min_products:float = 5) -> pd.DataFrame:
    """we want only orders with more than five products"""
    order_size = df.groupby("order_id")["outcome"].sum()
    orders_wanted = order_size[order_size >= min_products].index
    return df.loc[lambda x: x.order_id.isin(orders_wanted)]

def format_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    return(
        df
        .assign(created_at = lambda x: pd.to_datetime(x.created_at))
        .assign(order_date = lambda x: pd.to_datetime(x.order_date).dt.date)

    )


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLS], df[TARGET_COL]

def split_train_test(df:pd.DataFrame, test_size:float)-> tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
    daily_orders = df.groupby("order_date").order_id.nunique()
    cumsum_daily_orders = daily_orders.cumsum()/daily_orders.sum()

    train_cutoff = cumsum_daily_orders[cumsum_daily_orders <= test_size].idxmax()
    
    train_mask = df.order_date <= train_cutoff
    test_mask = df.order_date > train_cutoff

    X_train,y_train= split_features(df[train_mask])
    X_test,y_test = split_features(df[test_mask])

    return X_train,X_test,y_train,y_test


def save_model(model, model_name:str) -> None:
    path = MODEL_DIR/f"{model_name}.pkl"
    joblib.dump(model,path)
    logger.info(f"model saved at {path}")




def create_wanted_df() -> pd.DataFrame:
    logger.info("creating dataframe")
    return(
        load_dataset()
        .pipe(relevant_dataset)
        .pipe(format_date_cols)
        .sample(n=100000, random_state=42)
    )

def train_save_model(df: pd.DataFrame,name:str) -> None:
    logger.info(f"training Ridge model with C: {RIDGE_C}")

    X_train,X_test,y_train,y_test = split_train_test (df,TRAIN_PERCENT)

    model = make_pipeline(
        StandardScaler(),LogisticRegression(penalty="l2",C=RIDGE_C,)
    )

    model.fit(X_train,y_train)

    test_proba = model.predict_proba(X_test)[:,1]
    precision,recall,_ = precision_recall_curve(y_test,test_proba)
    pr_auc = auc(recall,precision)

    logger.info(f"Model trained. PR AUC = {pr_auc:.4f}")
    
    save_model(model,name)

















def main():
    dataframe = create_wanted_df()
    train_save_model(dataframe,"ridge_model")

if __name__ == "__main__":  
    main()  
