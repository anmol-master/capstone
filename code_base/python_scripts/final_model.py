# ======================================
# IMPORTS
# ======================================
import os
import getpass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

# ======================================
# LOAD DATA FROM POSTGRES (MULTI-TABLE)
# ======================================
# Configure Postgres connection via environment variables (recommended)
PG_HOST = os.getenv("PG_HOST", "db")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB   = os.getenv("PG_DB", "db")
PG_USER = os.getenv("PG_USER", "db_user")
PG_PASS = os.getenv("PG_PASS", "db_password")  # keep password out of code if possible
if PG_PASS is None:
    PG_PASS = getpass.getpass(prompt="Postgres password: ")

# Create SQLAlchemy engine
engine = create_engine(f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}")

# Discover tables whose name starts with 'transactions' (case-insensitive)
with engine.connect() as conn:
    res = conn.execute(text("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
          AND table_name ILIKE 'transactions%'
          AND table_schema NOT IN ('pg_catalog','information_schema');
    """))
    tables = [(row["table_schema"], row["table_name"]) for row in res]

if not tables:
    raise RuntimeError("No tables found with names starting with 'transactions' in the database.")

print(f"🔍 Found transaction tables: {[t[1] for t in tables]}")

# Read each table into a DataFrame, keep only rows where transaction_date is not null
datasets = {}
df_list = []
for schema, table in tables:
    full_table = f'"{schema}"."{table}"' if schema != 'public' else f'"{table}"'
    sql = f"SELECT * FROM {full_table} WHERE transaction_date IS NOT NULL"
    try:
        df = pd.read_sql_query(sql, engine)
        if "transaction_date" not in df.columns:
            raise KeyError(f"'transaction_date' column not found in table {table}")
        # normalize column names (optional)
        df.columns = [c.strip() for c in df.columns]
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        datasets[table] = df
        df_list.append(df)
        print(f"✅ Loaded {table}: {df.shape}")
    except Exception as e:
        print(f"⚠️ Skipped {table} due to error: {e}")

if not df_list:
    raise RuntimeError("No transaction data was loaded from any table.")

# Concatenate all discovered tables into a single 'transactions' DataFrame
transactions = pd.concat(df_list, ignore_index=True)
print(f"✅ Concatenated transactions DataFrame: {transactions.shape}")

# Quick sanity checks and column alignment
required_cols = {"transaction_date", "sku_id", "quantity_consumed", "total_cost"}
missing = required_cols - set(transactions.columns)
if missing:
    raise RuntimeError(f"Missing required columns in concatenated transactions: {missing}. "
                       "Adjust your table columns or mapping before proceeding.")

# Optional: restrict to the two years (if you want the exact original years)
# transactions = transactions[
#     transactions["transaction_date"].between("2023-01-01","2024-12-31")
# ].copy()

# ======================================
# ORIGINAL PIPELINE CONTINUES (unchanged)
# ======================================

# Top 30 SKUs by revenue
top_skus = transactions.groupby("sku_id")["total_cost"].sum().sort_values(ascending=False).head(30).index

# Aggregate daily consumption
sku_daily_top30 = (
    transactions[transactions["sku_id"].isin(top_skus)]
    .groupby(["sku_id", "transaction_date"])["quantity_consumed"]
    .sum()
    .reset_index()
)

# ======================================
# METRICS
# ======================================
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nrmse = rmse / (np.max(y_true) - np.min(y_true) + 1e-6)
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))
    wmape = 100 * np.sum(np.abs(y_pred - y_true)) / (np.sum(y_true) + 1e-6)
    return rmse, mae, nrmse, smape, wmape

# ======================================
# FEATURE ENGINEERING FOR XGBOOST
# ======================================
def create_features(series, n_lags=21):
    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags+1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["roll_mean_7"] = df["y"].rolling(7).mean()
    df["roll_std_7"] = df["y"].rolling(7).std()
    df = df.dropna()
    X = df.drop("y", axis=1).values
    y = df["y"].values
    return X, y

# ======================================
# FORECASTING FUNCTION (original: multiple models, train/test split)
# ======================================
def forecast_sku(ts, sku_id):
    results = {}
    preds = {}

    split_idx = int(len(ts)*0.9)
    train, test = ts[:split_idx], ts[split_idx:]
    train_points, test_points = len(train), len(test)

    # ETS
    try:
        ets_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=7)
        ets_fit = ets_model.fit()
        ets_pred = ets_fit.forecast(len(test))
        preds["ETS"] = ets_pred
        rmse, mae, nrmse, smape, wmape = evaluate(test, ets_pred)
        results["ETS"] = [rmse, mae, nrmse, smape, wmape, train_points, test_points]
    except:
        pass

    # ARIMA
    try:
        arima_model = ARIMA(train, order=(1,1,1))
        arima_fit = arima_model.fit()
        arima_pred = arima_fit.forecast(len(test))
        preds["ARIMA"] = arima_pred
        rmse, mae, nrmse, smape, wmape = evaluate(test, arima_pred)
        results["ARIMA"] = [rmse, mae, nrmse, smape, wmape, train_points, test_points]
    except:
        pass

    # Prophet
    try:
        df_prophet = ts.reset_index().rename(columns={"transaction_date":"ds","quantity_consumed":"y"})
        df_train = df_prophet.iloc[:split_idx]
        df_test = df_prophet.iloc[split_idx:]
        prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True)
        prophet_model.fit(df_train)
        future = prophet_model.make_future_dataframe(periods=len(df_test))
        forecast = prophet_model.predict(future)
        prophet_pred = forecast["yhat"].values[-len(df_test):]
        preds["Prophet"] = prophet_pred
        rmse, mae, nrmse, smape, wmape = evaluate(df_test["y"].values, prophet_pred)
        results["Prophet"] = [rmse, mae, nrmse, smape, wmape, train_points, test_points]
    except:
        pass

    # XGBoost
    try:
        X, y = create_features(ts)
        split_idx_adj = int(len(X)*0.9)
        X_train, X_test = X[:split_idx_adj], X[split_idx_adj:]
        y_train, y_test = y[:split_idx_adj], y[split_idx_adj:]
        xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        preds["XGBoost"] = xgb_pred
        rmse, mae, nrmse, smape, wmape = evaluate(y_test, xgb_pred)
        results["XGBoost"] = [rmse, mae, nrmse, smape, wmape, train_points, test_points]
    except:
        pass

    # Ensemble
    try:
        preds_list = []
        weights = []
        for model_name, metrics in results.items():
            nrmse_val = metrics[2]
            preds_list.append(preds[model_name])
            weights.append(1/(nrmse_val+1e-6))
        ensemble_pred = np.average(np.column_stack(preds_list), axis=1, weights=weights)
        preds["Ensemble"] = ensemble_pred
        rmse, mae, nrmse, smape, wmape = evaluate(test, ensemble_pred)
        results["Ensemble"] = [rmse, mae, nrmse, smape, wmape, train_points, test_points]
    except:
        pass

    df_results = pd.DataFrame([
        [sku_id, model, *metrics] for model, metrics in results.items()
    ], columns=["sku_id","model","rmse","mae","nrmse","smape","wmape","train_points","test_points"])

    return df_results, ts, test, preds.get("Ensemble", None), split_idx

# ======================================
# RUN FOR TOP 30 SKUs (original behavior)
# ======================================
all_results = []
plot_data = []   # for consolidated grid plotting

for sku in top_skus:
    ts = sku_daily_top30[sku_daily_top30["sku_id"]==sku].set_index("transaction_date")["quantity_consumed"]
    ts = ts.asfreq("D").fillna(0)
    res, ts_full, test, ensemble_pred, split_idx = forecast_sku(ts, sku)
    all_results.append(res)
    if ensemble_pred is not None:
        plot_data.append((sku, ts_full, test, ensemble_pred, split_idx))

results_df = pd.concat(all_results, ignore_index=True)
results_df = results_df.sort_values(["sku_id","nrmse"])
print("📊 Forecasting Results (Top SKUs):")

# ======================================
# GRID PLOTS (ALL SKUs)
# ======================================
num_skus = len(plot_data)
cols = 5
rows = int(np.ceil(num_skus / cols))
plt.figure(figsize=(20, rows * 3))

for i, (sku, ts, test, ensemble_pred, split_idx) in enumerate(plot_data, 1):
    plt.subplot(rows, cols, i)
    plt.plot(ts.index, ts.values, label="Actual", color="black", linewidth=1)
    plt.axvline(ts.index[split_idx], color="gray", linestyle="--")
    plt.plot(ts.index[-len(ensemble_pred):], ensemble_pred, label="Ensemble", color="magenta", linewidth=1.5)
    plt.title(f"SKU {sku}", fontsize=9)
    plt.xticks(rotation=30, fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()

plt.suptitle("📈 Forecast Comparison (Actual vs Ensemble) - Top SKUs", fontsize=14, y=1.02)
plt.legend(loc='upper right', fontsize=8)
plt.show()

# ======================================
# BEST MODEL SUMMARY
# ======================================
best_models = results_df.loc[results_df.groupby("sku_id")["nrmse"].idxmin()]
print("✅ Best model per SKU:")


# ======================================
# GRID PLOT FOR ALL SKUs (XGBoost only, with NRMSE)
# ======================================
import math

n_skus = len(top_skus)
cols = 5
rows = math.ceil(n_skus / cols)

fig, axes = plt.subplots(rows, cols, figsize=(22, rows * 3))
axes = axes.flatten()

for i, sku in enumerate(top_skus):
    ax = axes[i]
    ts = sku_daily_top30[sku_daily_top30["sku_id"] == sku].set_index("transaction_date")["quantity_consumed"].asfreq("D").fillna(0)

    # Train-test split
    split_idx = int(len(ts) * 0.9)
    train, test = ts[:split_idx], ts[split_idx:]

    # XGBoost Forecast
    X, y = create_features(ts)
    split_idx_adj = int(len(X) * 0.9)
    X_train, X_test = X[:split_idx_adj], X[split_idx_adj:]
    y_train, y_test = y[:split_idx_adj], y[split_idx_adj:]

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    rmse, mae, nrmse, smape, wmape = evaluate(y_test, y_pred)
    test_idx = ts.index[-len(y_pred):]

    # Plot actual vs predicted
    ax.plot(ts.index, ts.values, label="Actual", color="black", linewidth=1.8)
    ax.plot(test_idx, y_pred, label="XGBoost Pred", color="blue", linestyle="--", linewidth=1.8)
    ax.axvline(ts.index[split_idx], color="gray", linestyle="--", linewidth=1)

    # Title + metric text
    ax.set_title(f"SKU {sku}", fontsize=10)
    ax.text(0.02, 0.85, f"NRMSE: {nrmse:.3f}", transform=ax.transAxes,
            fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    ax.legend(fontsize=8)

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# ======================================
# NEW: ONE-MONTH FORECAST (KEEPING EXISTING CODE)
# ======================================
from datetime import timedelta

def forecast_sku_one_month(ts, sku_id, start_date="2025-01-01", end_date="2025-02-01"):
    """
    Train models on full history (using existing create_features etc.) and forecast daily values
    from start_date through end_date (inclusive). Returns a DataFrame with columns: sku_id, model, date, forecast
    This function is additive — it doesn't modify or remove any of the original behavior.
    """
    preds = {}

    forecast_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    forecast_days = len(forecast_dates)

    # Ensure series index is daily and sorted
    ts = ts.sort_index()
    ts = ts.asfreq("D").fillna(0)

    # ========== ETS ==========
    try:
        ets_model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=7)
        ets_fit = ets_model.fit()
        ets_pred = ets_fit.forecast(forecast_days)
        ets_pred.index = forecast_dates
        preds["ETS"] = ets_pred
    except Exception as e:
        print(f"[{sku_id}] ETS failed (one-month): {e}")

    # ========== ARIMA ==========
    try:
        arima_model = ARIMA(ts, order=(1,1,1))
        arima_fit = arima_model.fit()
        arima_pred = arima_fit.forecast(steps=forecast_days)
        arima_pred.index = forecast_dates
        preds["ARIMA"] = arima_pred
    except Exception as e:
        print(f"[{sku_id}] ARIMA failed (one-month): {e}")

    # ========== Prophet ==========
    try:
        df_prophet = ts.reset_index().rename(columns={"transaction_date": "ds", "quantity_consumed": "y"})
        prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True)
        prophet_model.fit(df_prophet)

        future = pd.DataFrame({"ds": forecast_dates})
        forecast = prophet_model.predict(future)
        prophet_pred = forecast.set_index("ds")["yhat"]
        preds["Prophet"] = prophet_pred
    except Exception as e:
        print(f"[{sku_id}] Prophet failed (one-month): {e}")

    # ========== XGBoost (recursive forecasting) ==========
    try:
        n_lags = 21
        # Prepare training features and fit on full history
        X, y = create_features(ts, n_lags=n_lags)
        # Only fit if we have enough rows after creating features
        if len(X) > 0:
            xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42)
            xgb_model.fit(X, y)

            # recursive forecast: use last n_lags from ts (most recent at end)
            hist = ts.values.tolist()
            future_preds = []

            for _ in range(forecast_days):
                last_values = hist[-n_lags:]
                # if not enough history pad with zeros at the left
                if len(last_values) < n_lags:
                    last_values = [0.0] * (n_lags - len(last_values)) + last_values

                # lag_1 should be most recent value, lag_2 previous, ...
                lags_for_model = last_values[::-1]  # now [lag_1, lag_2, ..., lag_n]

                # rolling stats on last up-to-7 values
                last_7 = hist[-7:]
                if len(last_7) < 7:
                    # pad if needed
                    last_7 = [0.0] * (7 - len(last_7)) + last_7
                roll_mean_7 = np.mean(last_7)
                roll_std_7 = np.std(last_7)

                features = np.array(lags_for_model + [roll_mean_7, roll_std_7]).reshape(1, -1)
                pred = xgb_model.predict(features)[0]
                # enforce non-negative forecasts for quantities
                pred = max(0.0, pred)
                future_preds.append(pred)
                hist.append(pred)

            preds["XGBoost"] = pd.Series(future_preds, index=forecast_dates)
        else:
            print(f"[{sku_id}] XGBoost skipped (one-month): not enough rows after creating features (n_lags={n_lags}).")
    except Exception as e:
        print(f"[{sku_id}] XGBoost failed (one-month): {e}")

    # ========== Ensemble (simple average of available model forecasts) ==========
    try:
        if preds:
            aligned = pd.concat(preds.values(), axis=1)
            aligned.columns = preds.keys()
            ensemble_pred = aligned.mean(axis=1)
            preds["Ensemble"] = ensemble_pred
    except Exception as e:
        print(f"[{sku_id}] Ensemble failed (one-month): {e}")

    # Build a combined results DataFrame for this SKU
    forecast_df = pd.DataFrame()
    for model, series in preds.items():
        df_temp = pd.DataFrame({
            "sku_id": sku_id,
            "model": model,
            "date": series.index,
            "forecast": series.values
        })
        forecast_df = pd.concat([forecast_df, df_temp], ignore_index=True)

    return forecast_df

# ======================================
# RUN ONE-MONTH FORECAST (adds to existing outputs; does not remove anything)
# ======================================
all_forecasts_one_month = []

for sku in top_skus:
    ts = sku_daily_top30[sku_daily_top30["sku_id"] == sku].set_index("transaction_date")["quantity_consumed"]
    ts = ts.asfreq("D").fillna(0)

    print(f"🔮 (One-month) Forecasting SKU {sku}...")
    forecast_df = forecast_sku_one_month(ts, sku)
    all_forecasts_one_month.append(forecast_df)

# Combine all one-month forecasts into one DataFrame
final_forecasts_one_month_df = pd.concat(all_forecasts_one_month, ignore_index=True)

print("✅ One-month forecasts generated for all SKUs!")
try:
    display(final_forecasts_one_month_df.head())
except Exception:
    print(final_forecasts_one_month_df.head())

# -----------------------------
# SAVE FORECASTS TO POSTGRES (UPSERT)
# -----------------------------
from sqlalchemy import text
from sqlalchemy.types import Date, String, Float, TIMESTAMP
from datetime import datetime

table_name = "sku_forecasts"
tmp_table = "tmp_sku_forecasts_upload"

# ensure date column is proper datetime/date type
final_forecasts_one_month_df["date"] = pd.to_datetime(final_forecasts_one_month_df["date"])

# 1) create target table if not exists (with unique constraint on sku_id,model,date)
create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
  id SERIAL PRIMARY KEY,
  sku_id TEXT NOT NULL,
  model TEXT NOT NULL,
  date DATE NOT NULL,
  forecast DOUBLE PRECISION,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE (sku_id, model, date)
);
"""
with engine.begin() as conn:
    conn.execute(text(create_table_sql))
    print(f"✅ Ensured target table {table_name} exists.")

# 2) write dataframe to a temporary table (replace if exists)
# Use SQLAlchemy's to_sql which will create the temp table for us
# Convert date to date only (Postgres DATE)
df_tmp = final_forecasts_one_month_df.copy()
df_tmp["date"] = df_tmp["date"].dt.date  # convert to python date

# Use to_sql (if your pandas/SQLAlchemy supports it). We use if_exists='replace' to create tmp table.
df_tmp.to_sql(tmp_table, engine, if_exists="replace", index=False,
              dtype={
                  "sku_id": String(),
                  "model": String(),
                  "date": Date(),
                  "forecast": Float()
              }, method="multi")
print(f"✅ Written temporary data to {tmp_table} ({len(df_tmp)} rows).")

# 3) upsert from tmp_table -> target table
upsert_sql = f"""
INSERT INTO {table_name} (sku_id, model, date, forecast, created_at, updated_at)
SELECT sku_id, model, date, forecast, NOW(), NOW() FROM {tmp_table}
ON CONFLICT (sku_id, model, date) DO UPDATE
  SET forecast = EXCLUDED.forecast,
      updated_at = EXCLUDED.updated_at;
"""
with engine.begin() as conn:
    res = conn.execute(text(upsert_sql))
    # res.rowcount may be DB-driver-dependent; printing simple success message
    print(f"✅ Upserted data from {tmp_table} into {table_name}.")

# 4) drop the temp table
drop_sql = f"DROP TABLE IF EXISTS {tmp_table};"
with engine.begin() as conn:
    conn.execute(text(drop_sql))
    print(f"🧹 Dropped temporary table {tmp_table}.")

print(f"✅ Forecasts saved to database table: {table_name}")
