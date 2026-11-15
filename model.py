"""Demand model: predicts units sold from price, discount, month, category and season."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import get_discount

FEATURES = ["category", "season", "price", "discount_pct", "month_sin", "month_cos", "offer_active"]


def build_features(df):
    X = pd.DataFrame({
        "category": df["category"].values,
        "season": df["season"].values,
        "price": df["price"].astype(float).values,
        "discount_pct": df["discount_pct"].astype(float).values,
        "month_sin": np.sin(2 * np.pi * df["month"].values / 12),
        "month_cos": np.cos(2 * np.pi * df["month"].values / 12),
    })
    X["offer_active"] = [int(get_discount(m, c, s)[0] > 0)
                         for m, c, s in zip(df["month"], df["category"], df["season"])]
    return X[FEATURES]


def train_model(sales):
    """Train and return (pipeline, metrics)."""
    X, y = build_features(sales), sales["units_sold"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["category", "season"])],
                            remainder="passthrough")
    pipe = Pipeline([("pre", pre),
                     ("gbr", GradientBoostingRegressor(n_estimators=250, max_depth=3,
                                                       learning_rate=0.08, random_state=42))])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    metrics = {"MAE": mean_absolute_error(y_te, pred), "R2": r2_score(y_te, pred), "rows": len(sales)}
    pipe.fit(X, y)  # refit on all data for serving
    return pipe, metrics


def predict_units(model, df):
    return np.clip(model.predict(build_features(df)), 0, None)


def predict_batch(model, df):
    """Add predicted units, revenue and profit columns to an uploaded dataframe."""
    out = df.copy()
    out["pred_units"] = predict_units(model, out).round(1)
    net = out["price"] * (1 - out["discount_pct"] / 100)
    out["pred_revenue"] = (out["pred_units"] * net).round(0)
    out["pred_profit"] = (out["pred_units"] * (net - out["cost"])).round(0)
    return out


def optimize_discount(model, product, month, max_discount=60, step=5):
    """Scan discounts and return (curve dataframe, best row by profit)."""
    discounts = list(range(0, max_discount + 1, step))
    grid = pd.DataFrame({
        "category": product["category"], "season": product["season"],
        "price": product["price"], "cost": product["cost"],
        "month": month, "discount_pct": discounts,
    })
    grid = predict_batch(model, grid)
    best = grid.loc[grid["pred_profit"].idxmax()]
    return grid, best
