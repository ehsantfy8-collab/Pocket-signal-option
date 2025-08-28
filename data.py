
import yfinance as yf
import pandas as pd
def fetch_candles(symbol: str, timeframe: str = "1m", lookback: int = 300) -> pd.DataFrame:
    tf_map = {"1m": ("1m", "7d"), "2m": ("2m", "60d"), "5m": ("5m", "60d"),
              "15m": ("15m", "60d"), "30m": ("30m", "60d"), "1h": ("60m", "730d"),
              "1d": ("1d", "max")}
    interval, period = tf_map.get(timeframe, ("1m","7d"))
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df.empty:
        raise RuntimeError("No data")
    df = df.rename(columns=str.lower)
    df = df[['open','high','low','close','volume']].dropna().tail(lookback)
    df.reset_index(inplace=True)
    return df
