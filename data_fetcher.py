"""
Live market data fetcher using yfinance.
Pulls options chains and OHLCV history for the dashboard.
"""

import numpy as nd
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from volatility_trading_algorithm import Option, BlackScholesModel, RealizedVolatilityEstimator


def get_spot_price(ticker: str) -> float:
    t = yf.Ticker(ticker)
    info = t.fast_info
    return float(info.last_price)


def get_risk_free_rate() -> float:
    """Approximate risk-free rate from 13-week T-bill yield."""
    try:
        tbill = yf.Ticker("^IRX")
        hist = tbill.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1]) / 100
    except Exception:
        pass
    return 0.05


def get_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    t = yf.Ticker(ticker)
    hist = t.history(period=period, interval="1d", auto_adjust=True)
    hist.index = hist.index.tz_localize(None)
    hist = hist.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    hist.index.name = "date"
    return hist.dropna()


def get_options_chain(ticker: str, spot: float, rfr: float) -> Tuple[pd.DataFrame, List[Option]]:
    """
    Fetch all available option expiries, build a unified DataFrame and
    a list of Option objects ready for VolatilitySurface.

    Returns:
        raw_df  – tidy DataFrame with columns:
                  strike, expiry_date, expiry_years, option_type,
                  mid_price, implied_vol, moneyness
        options – List[Option] for VolatilitySurface
    """
    t = yf.Ticker(ticker)
    exp_dates = t.options
    if not exp_dates:
        return pd.DataFrame(), []

    today = date.today()
    rows = []
    options: List[Option] = []

    for exp_str in exp_dates:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        expiry_years = max((exp_date - today).days / 365, 1 / 365)

        try:
            chain = t.option_chain(exp_str)
        except Exception:
            continue

        for opt_type, df in (("call", chain.calls), ("put", chain.puts)):
            for _, row in df.iterrows():
                strike = float(row["strike"])
                moneyness = strike / spot

                # Use mid-price; fall back to lastPrice if bid/ask missing
                bid = float(row.get("bid", 0) or 0)
                ask = float(row.get("ask", 0) or 0)
                last = float(row.get("lastPrice", 0) or 0)
                mid = (bid + ask) / 2 if (bid + ask) > 0 else last
                if mid <= 0:
                    continue

                # yfinance already provides IV; recalc only if 0
                iv = float(row.get("impliedVolatility", 0) or 0)
                if iv <= 0:
                    try:
                        iv = BlackScholesModel.implied_volatility(
                            mid, spot, strike, expiry_years, rfr, opt_type
                        )
                    except Exception:
                        continue
                if iv <= 0 or iv > 5:
                    continue

                rows.append({
                    "strike": strike,
                    "expiry_date": exp_str,
                    "expiry_years": expiry_years,
                    "option_type": opt_type,
                    "mid_price": mid,
                    "implied_vol": iv,
                    "moneyness": moneyness,
                })
                options.append(Option(
                    strike=strike,
                    expiry=expiry_years,
                    option_type=opt_type,
                    premium=mid,
                    underlying_price=spot,
                    implied_vol=iv,
                ))

    raw_df = pd.DataFrame(rows)
    return raw_df, options


def compute_realized_vols(hist: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add Parkinson, Garman-Klass, and Yang-Zhang RV columns to price history."""
    rv = RealizedVolatilityEstimator()
    o = hist["open"].values
    h = hist["high"].values
    l = hist["low"].values
    c = hist["close"].values

    hist = hist.copy()
    hist["rv_parkinson"] = rv.parkinson_volatility(h, l, window)
    hist["rv_gk"] = rv.garman_klass_volatility(o, h, l, c, window)
    hist["rv_yz"] = rv.yang_zhang_volatility(o, h, l, c, window)
    return hist


def get_atm_iv_series(ticker: str, spot: float, rfr: float) -> Optional[pd.Series]:
    """
    Build a rough ATM IV time series from the current options chain
    by looking at near-dated options closest to the money.
    Returns a single scalar (current ATM IV) wrapped in a Series for display.
    """
    t = yf.Ticker(ticker)
    exp_dates = t.options
    if not exp_dates:
        return None

    today = date.today()
    # pick the nearest expiry that's at least 7 days out
    target_exp = None
    for exp_str in exp_dates:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        if (exp_date - today).days >= 7:
            target_exp = exp_str
            break
    if target_exp is None:
        return None

    try:
        chain = t.option_chain(target_exp)
    except Exception:
        return None

    calls = chain.calls.copy()
    calls["moneyness"] = (calls["strike"] - spot).abs()
    atm_call = calls.sort_values("moneyness").iloc[0]
    iv = float(atm_call.get("impliedVolatility", 0) or 0)
    return iv if iv > 0 else None
