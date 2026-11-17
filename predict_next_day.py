"""
Next Gold Price Prediction
Loads the model trained by gold_prediction.py and estimates the price
for the next calendar day after the most recent date in the CSV.

Expected project structure:
gold_price_prediction/
    gold_prediction.py
    predict_next_day.py
    data/
        Gold Price.csv
    outputs/
        best_gold_model.pkl
        model_features.pkl
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Gold Price.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "best_gold_model.pkl"
FEATURES_PATH = OUTPUT_DIR / "model_features.pkl"
OUTPUT_CSV = OUTPUT_DIR / "next_gold_price_prediction.csv"

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


def main():
    print("=" * 65)
    print("GOLD PRICE PREDICTION - NEXT DAY ESTIMATE")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}\n"
            "Ensure Gold Price.csv is in the project's data folder."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}\n"
            "Run 'python gold_prediction.py' successfully first."
        )
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Saved feature list not found:\n{FEATURES_PATH}\n"
            "Run 'python gold_prediction.py' successfully first."
        )

    # Load the exact saved model and feature order.
    model = joblib.load(MODEL_PATH)
    saved_features = joblib.load(FEATURES_PATH)
    if list(saved_features) != FEATURES:
        FEATURES_LOCAL = list(saved_features)
    else:
        FEATURES_LOCAL = FEATURES

    # Read and clean the data consistently with the training script.
    df = pd.read_csv(DATA_PATH)
    if not {"Date", "Price"}.issubset(df.columns):
        raise ValueError(
            "The CSV must contain both 'Date' and 'Price' columns. "
            f"Found: {list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
    df["Price"] = (
        df["Price"].astype(str).str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False).str.strip()
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Date", "Price"])
    df = df[(df["Date"] >= "2014-01-01") & (df["Date"] <= "2025-12-31")]
    df = (
        df.sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    if len(df) < 31:
        raise ValueError(
            f"Only {len(df)} valid records found. At least 31 observations "
            "are required to calculate the historical features."
        )

    latest_date = df["Date"].iloc[-1]
    next_date = latest_date + pd.Timedelta(days=1)
    prices = df["Price"].astype(float).reset_index(drop=True)

    # Create the feature values for the next date from historical observations.
    next_features = {
        "Previous_Day_Price": float(prices.iloc[-1]),
        "Price_7_Observations_Ago": float(prices.iloc[-7]),
        "Price_30_Observations_Ago": float(prices.iloc[-30]),
        "Average_Last_7": float(prices.iloc[-7:].mean()),
        "Average_Last_30": float(prices.iloc[-30:].mean()),
        "Month": int(next_date.month),
        "Year": int(next_date.year),
        "Day_Of_Year": int(next_date.dayofyear),
    }

    feature_row = pd.DataFrame(
        [[next_features[feature] for feature in FEATURES_LOCAL]],
        columns=FEATURES_LOCAL,
    )

    predicted_price = float(model.predict(feature_row)[0])

    # Save prediction with the assumptions and latest source date.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame([{
        "Latest_Data_Date": latest_date.date().isoformat(),
        "Prediction_Date": next_date.date().isoformat(),
        "Latest_Observed_Price": float(prices.iloc[-1]),
        "Predicted_Gold_Price": predicted_price,
        "Model_File": MODEL_PATH.name,
        "Note": "Model estimate only; not a guaranteed market price.",
    }])
    result.to_csv(OUTPUT_CSV, index=False)

    print(f"Latest data date:       {latest_date.date()}")
    print(f"Next calendar date:     {next_date.date()}")
    print(f"Latest observed price:  {prices.iloc[-1]:,.2f}")
    print(f"Predicted gold price:   {predicted_price:,.2f}")
    print(f"\nSaved prediction to: {OUTPUT_CSV}")
    print("\nImportant: if the prediction date is a weekend or market holiday,")
    print("this is still a calendar-day estimate, not an official market quote.")


if __name__ == "__main__":
    main()
