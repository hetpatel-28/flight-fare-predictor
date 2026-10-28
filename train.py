"""Train a flight-fare model and save ONE pipeline (preprocessing + model).

Usage:
  python train.py                      # demo mode: synthetic data (clearly NOT real accuracy)
  python train.py --data itineraries.csv --sample 1000000   # real Expedia CSV, chunked read
"""
import argparse, json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CAT = ["startingAirport", "destinationAirport"]
NUM = ["isBasicEconomy", "isRefundable", "isNonStop"]
TARGET = "totalFare"


def load_real(path, sample, total_rows=82_000_000, chunksize=500_000):
    """Stream a huge CSV in chunks, keep needed columns only, downcast dtypes,
    and randomly sample ~`sample` rows (fraction = sample / total_rows per chunk)."""
    usecols = CAT + NUM + [TARGET]
    frac = min(1.0, sample / total_rows)
    parts, seen = [], 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk[NUM] = chunk[NUM].astype("int8")
        chunk[TARGET] = chunk[TARGET].astype("float32")
        parts.append(chunk.sample(frac=frac, random_state=42))
        seen += len(chunk)
    df = pd.concat(parts, ignore_index=True)
    print(f"Scanned {seen:,} rows, kept {len(df):,}")
    return df


def make_synthetic(n=20_000, seed=42):
    """Synthetic data WITH noise so metrics aren't trivially perfect. Demo only."""
    rng = np.random.default_rng(seed)
    ap = ["JFK", "LAX", "ORD", "DFW", "ATL", "MIA", "SFO", "BOS", "DEN", "SEA"]
    df = pd.DataFrame({
        "startingAirport": rng.choice(ap, n),
        "destinationAirport": rng.choice(ap, n),
        "isBasicEconomy": rng.integers(0, 2, n),
        "isRefundable": rng.integers(0, 2, n),
        "isNonStop": rng.integers(0, 2, n),
    })
    df = df[df.startingAirport != df.destinationAirport].reset_index(drop=True)
    route = df.startingAirport.map({a: i for i, a in enumerate(ap)}) * 17 + \
        df.destinationAirport.map({a: i for i, a in enumerate(ap)}) * 11
    df[TARGET] = (150 + route + 60 * df.isNonStop + 40 * df.isRefundable
                  - 50 * df.isBasicEconomy + rng.normal(0, 30, len(df)))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--sample", type=int, default=1_000_000)
    args = ap.parse_args()

    df = load_real(args.data, args.sample) if args.data else make_synthetic()
    mode = "real" if args.data else "synthetic-demo"
    print(f"Mode: {mode}, rows: {len(df):,}, memory: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")

    X, y = df[CAT + NUM], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = ColumnTransformer([("route", OneHotEncoder(handle_unknown="ignore"), CAT)],
                            remainder="passthrough")
    model = Pipeline([("pre", pre),
                      ("xgb", xgb.XGBRegressor(n_estimators=300, learning_rate=0.1,
                                               max_depth=6, subsample=0.8,
                                               n_jobs=-1, random_state=42))])
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    baseline = np.full_like(y_te, y_tr.mean(), dtype=float)
    metrics = {
        "mode": mode, "rows": int(len(df)),
        "mae": float(mean_absolute_error(y_te, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "r2": float(r2_score(y_te, pred)),
        "baseline_mae": float(mean_absolute_error(y_te, baseline)),
    }
    print(json.dumps(metrics, indent=2))

    joblib.dump(model, "model.pkl")
    json.dump({"metrics": metrics,
               "known_airports": sorted(set(df.startingAirport.astype(str)) |
                                        set(df.destinationAirport.astype(str)))},
              open("model_meta.json", "w"), indent=2)
    print("Saved model.pkl and model_meta.json")


if __name__ == "__main__":
    main()