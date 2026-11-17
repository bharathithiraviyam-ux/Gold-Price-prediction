"""
Gold Price Prediction Using Machine Learning
Complete training, evaluation, plotting, and model-saving script.

Expected project structure:
gold_price_prediction/
    gold_prediction.py
    data/
        Gold Price.csv
    outputs/  (created automatically)

Expected CSV columns include: Date, Price
Other columns such as Open, High, Low, Volume, Chg% are optional.
"""

from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # Save plots without opening blocking windows.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# 1. Paths and configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Gold Price.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2014-01-01"
END_DATE = "2025-12-31"
TEST_FRACTION = 0.20

print("=" * 65)
print("GOLD PRICE PREDICTION - MACHINE LEARNING")
print("=" * 65)

# -----------------------------
# 2. Load and validate dataset
# -----------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}\n"
        "Make sure 'Gold Price.csv' is inside the project's data folder."
    )

df = pd.read_csv(DATA_PATH)

required_columns = {"Date", "Price"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(
        f"CSV is missing required column(s): {', '.join(sorted(missing_columns))}. "
        f"Found columns: {list(df.columns)}"
    )

df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
# Support common price formatting such as commas or currency symbols.
df["Price"] = (
    df["Price"].astype(str).str.replace(",", "", regex=False)
    .str.replace("$", "", regex=False).str.strip()
)
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

df = df.dropna(subset=["Date", "Price"])
df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]
df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

if len(df) < 100:
    raise ValueError(
        f"Only {len(df)} usable records found for 2014–2025. "
        "Please check the Date format and dataset contents."
    )

print(f"Dataset: {DATA_PATH}")
print(f"Usable records (2014–2025): {len(df):,}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Columns: {list(df.columns)}")

# Save a cleaned copy for reproducibility.
df.to_csv(OUTPUT_DIR / "cleaned_gold_prices_2014_2025.csv", index=False)

# -----------------------------
# 3. Exploratory data analysis
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Price"], linewidth=1.2)
plt.title("Gold Price Over Time (2014–2025)")
plt.xlabel("Date")
plt.ylabel("Gold Price")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gold_price_over_time.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(9, 5))
plt.hist(df["Price"], bins=40, edgecolor="black")
plt.title("Distribution of Gold Prices (2014–2025)")
plt.xlabel("Gold Price")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gold_price_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# 4. Feature engineering
# -----------------------------
# These are based only on past observations to avoid target leakage.
df["Previous_Day_Price"] = df["Price"].shift(1)
df["Price_7_Observations_Ago"] = df["Price"].shift(7)
df["Price_30_Observations_Ago"] = df["Price"].shift(30)
past_prices = df["Price"].shift(1)
df["Average_Last_7"] = past_prices.rolling(window=7).mean()
df["Average_Last_30"] = past_prices.rolling(window=30).mean()
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year
df["Day_Of_Year"] = df["Date"].dt.dayofyear

FEATURES = [
    "Previous_Day_Price",
    "Price_7_Observations_Ago",
    "Price_30_Observations_Ago",
    "Average_Last_7",
    "Average_Last_30",
    "Month",
    "Year",
    "Day_Of_Year",
]

model_df = df.dropna(subset=FEATURES + ["Price"]).reset_index(drop=True)
X = model_df[FEATURES]
y = model_df["Price"]
dates = model_df["Date"]

if len(model_df) < 100:
    raise ValueError("Not enough rows remain after creating historical features.")

# -----------------------------
# 5. Chronological train/test split
# -----------------------------
split_index = int(len(model_df) * (1 - TEST_FRACTION))
if split_index <= 0 or split_index >= len(model_df):
    raise ValueError("Invalid train/test split. Check dataset size.")

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
test_dates = dates.iloc[split_index:]

print("\n" + "=" * 65)
print("DATA SPLIT")
print("=" * 65)
print(f"Features: {FEATURES}")
print(f"Training records: {len(X_train):,}")
print(f"Testing records: {len(X_test):,}")
print(f"Training period: {dates.iloc[0].date()} to {dates.iloc[split_index - 1].date()}")
print(f"Testing period: {test_dates.iloc[0].date()} to {test_dates.iloc[-1].date()}")

# -----------------------------
# 6. Train models
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
}

predictions = {}
result_rows = []

print("\n" + "=" * 65)
print("TRAINING MODELS")
print("=" * 65)

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    result_rows.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2,
    })

results_df = pd.DataFrame(result_rows).sort_values("MAE").reset_index(drop=True)

# -----------------------------
# 7. Baseline evaluation
# -----------------------------
# Baseline: next observed price is predicted to equal the previous observed price.
baseline_predictions = X_test["Previous_Day_Price"].to_numpy()
baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_predictions)))
baseline_r2 = r2_score(y_test, baseline_predictions)

baseline_df = pd.DataFrame([{
    "Model": "Previous-price baseline",
    "MAE": baseline_mae,
    "RMSE": baseline_rmse,
    "R2 Score": baseline_r2,
}])

print("\n" + "=" * 65)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 65)
print(results_df.to_string(index=False, formatters={
    "MAE": "{:.2f}".format,
    "RMSE": "{:.2f}".format,
    "R2 Score": "{:.4f}".format,
}))

print("\nBASELINE PERFORMANCE")
print(f"Baseline MAE:  {baseline_mae:.2f}")
print(f"Baseline RMSE: {baseline_rmse:.2f}")
print(f"Baseline R²:   {baseline_r2:.4f}")

best_model_name = str(results_df.iloc[0]["Model"])
best_model = models[best_model_name]
best_model_mae = float(results_df.iloc[0]["MAE"])

if best_model_mae < baseline_mae:
    print(f"\n{best_model_name} has lower MAE than the baseline on this test period.")
else:
    print(
        "\nWARNING: The best trained model does not beat the previous-price baseline "
        "on MAE for this test period. Treat its forecasts cautiously."
    )

# -----------------------------
# 8. Save predictions and metrics
# -----------------------------
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
baseline_df.to_csv(OUTPUT_DIR / "baseline_comparison.csv", index=False)

prediction_output = pd.DataFrame({
    "Date": test_dates.to_numpy(),
    "Actual_Price": y_test.to_numpy(),
    "Baseline_Prediction": baseline_predictions,
})
for model_name, y_pred in predictions.items():
    safe_name = model_name.replace(" ", "_")
    prediction_output[f"{safe_name}_Prediction"] = y_pred

prediction_output.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

# -----------------------------
# 9. Save actual vs predicted graph
# -----------------------------
plt.figure(figsize=(14, 7))
plt.plot(test_dates, y_test.to_numpy(), label="Actual Price", linewidth=2)
for model_name, y_pred in predictions.items():
    plt.plot(test_dates, y_pred, label=f"{model_name} Prediction", alpha=0.85)
plt.title("Actual vs Predicted Gold Prices (Test Period)")
plt.xlabel("Date")
plt.ylabel("Gold Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot the best model separately for easier presentation.
plt.figure(figsize=(14, 7))
plt.plot(test_dates, y_test.to_numpy(), label="Actual Price", linewidth=2)
plt.plot(test_dates, predictions[best_model_name], label=f"{best_model_name} Prediction", linewidth=1.5)
plt.title(f"Actual vs {best_model_name} Prediction")
plt.xlabel("Date")
plt.ylabel("Gold Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "best_model_prediction.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# 10. Save model and metadata
# -----------------------------
joblib.dump(best_model, OUTPUT_DIR / "best_gold_model.pkl")
joblib.dump(FEATURES, OUTPUT_DIR / "model_features.pkl")
joblib.dump(
    {
        "model_name": best_model_name,
        "features": FEATURES,
        "training_start_date": str(dates.iloc[0].date()),
        "training_end_date": str(dates.iloc[split_index - 1].date()),
        "testing_start_date": str(test_dates.iloc[0].date()),
        "testing_end_date": str(test_dates.iloc[-1].date()),
        "test_mae": best_model_mae,
        "test_rmse": float(results_df.iloc[0]["RMSE"]),
        "test_r2": float(results_df.iloc[0]["R2 Score"]),
    },
    OUTPUT_DIR / "model_metadata.pkl",
)

# Save model results summary as a readable text file.
with open(OUTPUT_DIR / "training_summary.txt", "w", encoding="utf-8") as file:
    file.write("GOLD PRICE PREDICTION - TRAINING SUMMARY\n")
    file.write("=" * 50 + "\n")
    file.write(f"Dataset records: {len(df)}\n")
    file.write(f"Usable model records: {len(model_df)}\n")
    file.write(f"Best model by test MAE: {best_model_name}\n")
    file.write(f"Best model MAE: {best_model_mae:.4f}\n")
    file.write(f"Best model RMSE: {float(results_df.iloc[0]['RMSE']):.4f}\n")
    file.write(f"Best model R²: {float(results_df.iloc[0]['R2 Score']):.6f}\n")
    file.write(f"Baseline MAE: {baseline_mae:.4f}\n")
    file.write(f"Baseline RMSE: {baseline_rmse:.4f}\n")
    file.write(f"Baseline R²: {baseline_r2:.6f}\n")
    file.write("\nModel comparison:\n")
    file.write(results_df.to_string(index=False))
    file.write("\n\nBaseline comparison:\n")
    file.write(baseline_df.to_string(index=False))
    file.write("\n")

print("\n" + "=" * 65)
print("TRAINING COMPLETE")
print("=" * 65)
print(f"Best model by MAE: {best_model_name}")
print(f"Best model MAE: {best_model_mae:.2f}")
print(f"Saved model: {OUTPUT_DIR / 'best_gold_model.pkl'}")
print(f"Saved feature list: {OUTPUT_DIR / 'model_features.pkl'}")
print(f"All outputs saved in: {OUTPUT_DIR}")
print("\nGenerated files:")
for output_file in sorted(OUTPUT_DIR.iterdir()):
    if output_file.is_file():
        print(f" - {output_file.name}")
print("\nNote: This is a statistical estimate, not a guarantee of future gold prices.")
