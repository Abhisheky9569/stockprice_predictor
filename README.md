# Stock Price Predictor

A beginner-friendly machine-learning project that downloads historical market data, trains a Linear Regression model, evaluates it on a later time period, and saves a chart and CSV of predictions.

## Features
- Downloads adjusted daily prices using `yfinance`
- Uses a chronological train/test split to avoid shuffling time-series observations
- Trains a Linear Regression baseline
- Reports MAE, RMSE, and R²
- Saves test predictions and a chart in `outputs/`

## Setup

Python 3.10+ recommended.

```bash
git clone <your-repository-url>
cd stock-price-predictor
python -m venv .venv
```

Activate the environment:
- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies and run:

```bash
pip install -r requirements.txt
python stock_predictor.py --ticker AAPL --period 2y
```

Other examples:

```bash
python stock_predictor.py --ticker MSFT --period 5y
python stock_predictor.py --ticker RELIANCE.NS --period 2y
```

## Outputs
The script creates an `outputs/` folder containing:
- `<TICKER>_predictions.csv`
- `<TICKER>_prediction_chart.png`

## Method
The model uses the previous trading day's closing price to estimate the current day's close. The final 20% of observations are held out as the test set. This is intentionally a simple baseline and is not a robust trading strategy.

## Limitations
- Historical prices alone omit company fundamentals, news, market conditions, and transaction costs.
- A one-feature linear regression model may perform poorly and can fail to beat a naive baseline.
- The next-step estimate is illustrative, not a reliable forecast.
- This project is for learning only, not financial advice.

## Possible improvements
Add lag features, moving averages, volume, walk-forward validation, naive-baseline comparisons, and models such as Random Forest or gradient boosting.
