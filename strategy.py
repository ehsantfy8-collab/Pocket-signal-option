
import pandas as pd
from indicators import ema, rsi, macd, bbands, stoch, atr, adx, heikin_ashi, engulfing, pinbar
def analyze_symbol(symbol: str, df: pd.DataFrame, timeframe: str):
    c = df['close']; h = df['high']; l = df['low']; o = df['open']
    ema_fast = ema(c, 9); ema_slow = ema(c, 21)
    rsi14 = rsi(c, 14); macd_line, macd_sig, macd_hist = macd(c)
    bb_l, bb_m, bb_u, bb_w = bbands(c, 20, 2.0)
    st_k, st_d = stoch(h, l, c, 14, 3)
    atr14 = atr(h, l, c, 14); adx14 = adx(h, l, c, 14)
    ha = heikin_ashi(df); eng = engulfing(df); pin = pinbar(df)
    last = len(df)-1
    trend_up = ema_fast.iloc[last] > ema_slow.iloc[last]
    trend_down = ema_fast.iloc[last] < ema_slow.iloc[last]
    overbought = rsi14.iloc[last] >= 70; oversold = rsi14.iloc[last] <= 30
    bb_touch_low = c.iloc[last] <= bb_l.iloc[last]; bb_touch_up = c.iloc[last] >= bb_u.iloc[last]
    st_oversold = st_k.iloc[last] < 20 and st_d.iloc[last] < 20
    st_overbought = st_k.iloc[last] > 80 and st_d.iloc[last] > 80
    eng_sig = eng.iloc[last]; pin_sig = pin.iloc[last]
    macd_bull = macd_hist.iloc[last] > 0; macd_bear = macd_hist.iloc[last] < 0
    strong_trend = (adx14.iloc[last] or 0) > 20
    bull_score = 0; bear_score = 0
    if trend_up: bull_score += 2
    if trend_down: bear_score += 2
    if oversold: bull_score += 2
    if overbought: bear_score += 2
    if bb_touch_low: bull_score += 1
    if bb_touch_up: bear_score += 1
    if st_oversold: bull_score += 1
    if st_overbought: bear_score += 1
    if eng_sig == 1: bull_score += 1
    if eng_sig == -1: bear_score += 1
    if pin_sig == 1: bull_score += 1
    if pin_sig == -1: bear_score += 1
    if macd_bull: bull_score += 1
    if macd_bear: bear_score += 1
    if strong_trend and trend_up: bull_score += 1
    if strong_trend and trend_down: bear_score += 1
    if bull_score - bear_score >= 3:
        signal = "CALL (BUY)"; quality = "strong"
    elif bear_score - bull_score >= 3:
        signal = "PUT (SELL)"; quality = "strong"
    elif bull_score > bear_score:
        signal = "CALL (BUY)"; quality = "weak"
    elif bear_score > bull_score:
        signal = "PUT (SELL)"; quality = "weak"
    else:
        signal = "NO TRADE"; quality = "neutral"
    text = (f"نماد: {symbol}\n"
            f"تایم‌فریم: {timeframe}\n"
            f"سیگنال: {signal} | کیفیت: {quality}\n"
            f"RSI(14): {rsi14.iloc[last]:.1f} | ADX: {adx14.iloc[last]:.1f}\n"
            f"EMA9 {'>' if trend_up else '<'} EMA21 | MACD Hist: {macd_hist.iloc[last]:.4f}\n"
            f"BBand touch: {'Lower' if bb_touch_low else ('Upper' if bb_touch_up else '-')} | Stoch: {int(st_k.iloc[last])}/{int(st_d.iloc[last])}\n"
            f"Candle: {'Engulfing Bull' if eng_sig==1 else ('Engulfing Bear' if eng_sig==-1 else '-')} / "
            f"{'Pin Bull' if pin_sig==1 else ('Pin Bear' if pin_sig==-1 else '-')}\n"
            f"امتیاز Bull/Bear: {bull_score}/{bear_score}")
    return {"signal": signal, "quality": quality, "bull_score": int(bull_score), "bear_score": int(bear_score),
            "rsi": float(rsi14.iloc[last]), "adx": float(adx14.iloc[last]) if pd.notna(adx14.iloc[last]) else 0.0, "text": text}
