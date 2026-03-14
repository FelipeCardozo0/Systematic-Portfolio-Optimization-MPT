"""Data downloading and preprocessing module."""
import numpy as np
import pandas as pd
import yfinance as yf


def download_prices(tickers, start_date, end_date):
    """Download adjusted close prices for given tickers."""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    if len(tickers) == 1:
        prices = data['Adj Close'].to_frame()
        prices.columns = tickers
    else:
        prices = data['Adj Close']
    return prices.dropna()


def compute_returns(prices):
    """Compute daily log returns from prices."""
    return np.log(prices / prices.shift(1)).dropna()


def annualize_params(returns, trading_days=252):
    """Compute annualized mean returns and covariance matrix."""
    daily_returns = returns.mean()
    daily_cov = returns.cov()

    mu = daily_returns * trading_days
    sigma = daily_cov * trading_days

    return mu.values, sigma.values

