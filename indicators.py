
import pandas as pd
import numpy as np
def sma(s, n): return s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(series, length=14):
    delta = series.diff()
    up = (delta.where(delta > 0, 0)).rolling(length).mean()
    down = (-delta.where(delta < 0, 0)).rolling(length).mean()
    rs = up / (down.replace(0, np.nan))
    out = 100 - (100/(1+rs))
    return out.fillna(50)
def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist
def bbands(series, length=20, mult=2.0):
    mid = sma(series, length)
    std = series.rolling(length).std()
    upper = mid + mult*std
    lower = mid - mult*std
    width = (upper - lower) / mid
    return lower, mid, upper, width
def stoch(high, low, close, k=14, d=3):
    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()
    percent_k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    percent_d = percent_k.rolling(d).mean()
    return percent_k.fillna(50), percent_d.fillna(50)
def atr(high, low, close, n=14):
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()
def adx(high, low, close, n=14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = atr(high, low, close, n)
    plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(n).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(n).mean() / tr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace(0, np.nan) * 100
    return dx.rolling(n).mean()
def heikin_ashi(df):
    ha = df.copy()
    ha['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = [ (df['open'].iloc[0] + df['close'].iloc[0]) / 2 ]
    for i in range(1, len(df)):
        ha_open.append( (ha_open[i-1] + ha['ha_close'].iloc[i-1]) / 2 )
    ha['ha_open'] = pd.Series(ha_open, index=df.index)
    ha['ha_high'] = ha[['high','ha_open','ha_close']].max(axis=1)
    ha['ha_low']  = ha[['low','ha_open','ha_close']].min(axis=1)
    return ha[['ha_open','ha_high','ha_low','ha_close']]
def engulfing(df):
    prev = df.shift(1)
    bull = (df['close'] > df['open']) & (prev['close'] < prev['open']) & (df['close'] >= prev['open']) & (df['open'] <= prev['close'])
    bear = (df['close'] < df['open']) & (prev['close'] > prev['open']) & (df['open'] >= prev['open']) & (df['close'] <= prev['close'])
    return bull.astype(int) - bear.astype(int)
def pinbar(df, thresh=0.66):
    body = (df['close'] - df['open']).abs()
    rng = (df['high'] - df['low']).replace(0, np.nan)
    upper_wick = df['high'] - df[['open','close']].max(axis=1)
    lower_wick = df[['open','close']].min(axis=1) - df['low']
    bull = (lower_wick / rng > thresh) & (body / rng < (1 - thresh)/2)
    bear = (upper_wick / rng > thresh) & (body / rng < (1 - thresh)/2)
    return bull.astype(int) - bear.astype(int)
