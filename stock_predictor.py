"""
Stock Price Predictor
Educational demo using historical daily closing prices and Linear Regression.
This is not investment advice; stock prices are inherently difficult to predict.
"""
from pathlib import Path
import argparse
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def download_data(ticker: str, period: str) -> pd.DataFrame:
    """Download historical stock prices from Yahoo Finance."""
    data = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. Check the symbol and your internet connection."
        )
    # yfinance may return either a flat or a MultiIndex column structure.
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    result = close.rename("Close").dropna().to_frame()
    result.index = pd.to_datetime(result.index)
    return result


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Use the previous day's close to predict the next trading day's close."""
    df = data.copy()
    df["Previous_Close"] = df["Close"].shift(1)
    df["Target"] = df["Close"]
    return df.dropna()


def train_and_evaluate(ticker: str = "AAPL", period: str = "2y") -> None:
    data = download_data(ticker, period)
    dataset = build_features(data)

    if len(dataset) < 30:
        raise ValueError("Not enough historical data. Try a longer period such as '2y' or '5y'.")

    X = dataset[["Previous_Close"]]
    y = dataset["Target"]

    # Keep time order; do not shuffle time-series data.
    split = int(len(dataset) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"\nStock Price Predictor — {ticker.upper()}")
    print("=" * 42)
    print(f"Historical period: {period}")
    print(f"Training rows:     {len(X_train)}")
    print(f"Testing rows:      {len(X_test)}")
    print(f"MAE:               {mae:.4f}")
    print(f"RMSE:              {rmse:.4f}")
    print(f"R²:                {r2:.4f}")
    print("\nNote: These metrics do not guarantee future accuracy and this is not financial advice.")

    results = pd.DataFrame({
        "Actual_Close": y_test,
        "Predicted_Close": predictions
    }, index=y_test.index)
    results.to_csv(OUTPUT_DIR / f"{ticker.upper()}_predictions.csv", index_label="Date")

    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results["Actual_Close"], label="Actual closing price")
    plt.plot(results.index, results["Predicted_Close"], label="Predicted closing price")
    plt.title(f"{ticker.upper()} — Actual vs Predicted Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Adjusted closing price")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    chart_path = OUTPUT_DIR / f"{ticker.upper()}_prediction_chart.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    # One-step estimate based on the latest observed close.
    latest_close = float(data["Close"].iloc[-1])
    next_estimate = float(model.predict(pd.DataFrame({"Previous_Close": [latest_close]}))[0])
    print(f"\nLatest observed close: {latest_close:.2f}")
    print(f"Illustrative next-step estimate: {next_estimate:.2f}")
    print(f"Saved predictions: {OUTPUT_DIR / f'{ticker.upper()}_predictions.csv'}")
    print(f"Saved chart:       {chart_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict stock closing prices with Linear Regression.")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker, e.g. AAPL, MSFT, RELIANCE.NS")
    parser.add_argument("--period", default="2y", help="Yahoo Finance period, e.g. 1y, 2y, 5y")
    args = parser.parse_args()
    try:
        train_and_evaluate(args.ticker, args.period)
    except Exception as exc:
        raise SystemExit(f"Error: {exc}")
