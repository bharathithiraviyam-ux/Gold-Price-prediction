
from pathlib import Path
from datetime import date, timedelta

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Gold Price.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "best_gold_model.pkl"
FEATURES_PATH = OUTPUT_DIR / "model_features.pkl"
METRICS_PATH = OUTPUT_DIR / "model_comparison.csv"

EXPECTED_FEATURES = [
    "Previous_Day_Price",
    "Price_7_Observations_Ago",
    "Price_30_Observations_Ago",
    "Average_Last_7",
    "Average_Last_30",
    "Month",
    "Year",
    "Day_Of_Year",
]

st.set_page_config(
    page_title="Gold Price Prediction",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Gold Price Prediction Using Machine Learning")
st.write("Choose a future date to estimate the gold price.")
st.warning(
    "This is an experimental model estimate, not a live market price "
    "or financial advice."
)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Check the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    if not {"Date", "Price"}.issubset(df.columns):
        raise ValueError("CSV must contain Date and Price columns.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = (
        df["Price"].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df = df.dropna(subset=["Date", "Price"])
    df = df[
        (df["Date"] >= "2014-01-01")
        & (df["Date"] <= "2025-12-31")
    ]
    df = (
        df.sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    if len(df) < 31:
        raise ValueError("At least 31 valid price records are required.")

    return df


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Trained model files are missing. Run "
            "python gold_prediction.py successfully first."
        )

    model = joblib.load(MODEL_PATH)
    features = list(joblib.load(FEATURES_PATH))

    if features != EXPECTED_FEATURES:
        raise ValueError(
            "The saved model features do not match this app. "
            "Retrain the model using the latest gold_prediction.py."
        )

    return model, features


def make_feature_row(target_date, prices, features):
    # The same feature definitions used in model training.
    values = {
        "Previous_Day_Price": float(prices[-1]),
        "Price_7_Observations_Ago": float(prices[-7]),
        "Price_30_Observations_Ago": float(prices[-30]),
        "Average_Last_7": float(sum(prices[-7:]) / 7),
        "Average_Last_30": float(sum(prices[-30:]) / 30),
        "Month": target_date.month,
        "Year": target_date.year,
        "Day_Of_Year": target_date.timetuple().tm_yday,
    }

    return pd.DataFrame(
        [[values[name] for name in features]],
        columns=features,
    )


def forecast_to_date(model, features, df, target_date):
    # Future prices are unknown, so each prediction becomes
    # an estimated input for the next calendar day.
    prices = df["Price"].astype(float).tolist()
    latest_date = pd.Timestamp(df["Date"].iloc[-1]).normalize()
    target_date = pd.Timestamp(target_date).normalize()

    if target_date <= latest_date:
        raise ValueError(
            f"Choose a date after {latest_date.date()}."
        )

    forecast_dates = pd.date_range(
        start=latest_date + pd.Timedelta(days=1),
        end=target_date,
        freq="D",
    )

    predictions = []

    for forecast_date in forecast_dates:
        row = make_feature_row(forecast_date, prices, features)
        prediction = float(model.predict(row)[0])

        if not pd.notna(prediction):
            raise ValueError("The model returned an invalid prediction.")

        predictions.append({
            "Date": forecast_date,
            "Predicted_Price": prediction,
        })
        prices.append(prediction)

    forecast_df = pd.DataFrame(predictions)
    return latest_date, float(forecast_df.iloc[-1]["Predicted_Price"]), forecast_df


try:
    df = load_data()
    model, features = load_model()
except Exception as error:
    st.error(str(error))
    st.stop()

latest_date = pd.Timestamp(df["Date"].iloc[-1]).date()
first_future_date = latest_date + timedelta(days=1)

st.caption(
    f"Dataset dates: {df['Date'].min().date()} to {latest_date}. "
    f"Earliest selectable future date: {first_future_date}."
)

st.subheader("Select a Future Date")

selected_date = st.date_input(
    "Date for gold-price prediction",
    value=first_future_date,
    min_value=first_future_date,
    max_value=date(2040, 12, 31),
)

forecast_days = (selected_date - latest_date).days
st.write(f"Forecast horizon: **{forecast_days} calendar days**")

if st.button("Predict Gold Price", type="primary"):
    try:
        with st.spinner("Generating prediction..."):
            origin_date, predicted_price, forecast_df = forecast_to_date(
                model, features, df, selected_date
            )

        st.success("Prediction generated.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Selected Date", selected_date.strftime("%d %b %Y"))
        col2.metric("Estimated Gold Price", f"{predicted_price:,.2f}")
        col3.metric("Latest Recorded Price", f"{df['Price'].iloc[-1]:,.2f}")

        st.caption(
            f"Forecast begins after {origin_date.date()}. "
            f"It uses {len(forecast_df)} recursive calendar-day steps."
        )

        st.subheader("Historical Prices and Forecast")

        historical = df[["Date", "Price"]].tail(365).copy()
        historical = historical.rename(columns={"Price": "Historical Price"})
        historical = historical.set_index("Date")

        forecast_chart = forecast_df.rename(
            columns={"Predicted_Price": "Forecast Price"}
        ).set_index("Date")

        chart = pd.concat(
            [historical[["Historical Price"]], forecast_chart],
            axis=1,
        )
        st.line_chart(chart, use_container_width=True)

        with st.expander("View forecast values"):
            st.dataframe(
                forecast_df.rename(
                    columns={"Predicted_Price": "Predicted Gold Price"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            "Download Forecast CSV",
            data=forecast_df.to_csv(index=False).encode("utf-8"),
            file_name=f"gold_forecast_{selected_date.isoformat()}.csv",
            mime="text/csv",
        )

        if forecast_days > 30:
            st.warning(
                "This date is more than 30 days ahead. "
                "Recursive prediction errors can accumulate substantially."
            )
        elif forecast_days > 7:
            st.info(
                "This multi-day forecast may be less accurate than "
                "a one-step-ahead prediction."
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


st.divider()
st.subheader("Model Evaluation")

metrics_path = OUTPUT_DIR / "model_comparison.csv"
if metrics_path.exists():
    st.dataframe(
        pd.read_csv(metrics_path).round(4),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Run gold_prediction.py to create model_comparison.csv.")

st.subheader("How It Works")
st.markdown(
    """
    1. Reads historical gold prices.
    2. Loads the trained regression model.
    3. Creates the eight features used during training.
    4. Predicts each future calendar day in sequence.
    5. Uses each estimated price to construct features for the next day.
    6. Displays the estimate for the date selected by the user.
    """
)

st.warning(
    "Limitations: the model was trained on historical data ending in 2025. "
    "It does not know future news, interest rates, currency movements, or "
    "market conditions. For future dates, it repeatedly uses its own "
    "predictions as inputs, so errors may compound. Calendar-day steps "
    "also include weekends and market holidays. Treat this as an academic "
    "demonstration, not a dependable market forecast."
)
