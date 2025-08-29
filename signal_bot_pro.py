#!/usr/bin/env python3
"""
Telegram Signal Bot PRO — آموزشی (بدون اتصال به بروکر و بدون هرگونه سود تضمینی)

🔧 چی اضافه شده؟
- استراتژی‌ها: ema_rsi | macd | donchian | supertrend | bb_meanrev
- فیلترها: regime (EMA200) و volatility (ATR)
- رای‌گیری (ensemble): /ensemble on|off  و وزن‌دهی ساده
- سیگنال‌های «قوی/ضعیف» بر اساس فیلترها
- همان دستورات نسخه ساده + /filters و /ensemble

⚠️ هشدار: هیچ سودی تضمین نیست. این ابزار آموزشی است و به هیچ بروکری وصل نمی‌شود.
"""

from __future__ import annotations
import os, io, json, asyncio, logging, time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("signal-bot-pro")

# ---------- Config ----------
BINANCE_BASE = "https://api.binance.com"
VALID_INTERVALS = {"1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"}

# ---------- Indicators ----------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100/(1+rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l).abs(), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

# ---------- Strategies ----------
@dataclass
class Params:
    strategy: str = "ema_rsi"        # ema_rsi | macd | donchian | supertrend | bb_meanrev
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_len: int = 14
    rsi_buy: int = 55
    rsi_sell: int = 45
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    donchian_len: int = 20
    st_period: int = 10
    st_mult: float = 3.0
    bb_len: int = 20
    bb_k: float = 2.0
    lookback: int = 400

@dataclass
class Filters:
    regime_ema: int = 200     # >0 فعال؛ 0 = غیرفعال
    vol_atr: int = 14         # >0 فعال؛ 0 = غیرفعال
    min_atr_frac: float = 0.001  # حداقل ATR به نسبت قیمت (برای حذف بازار خیلی کم‌نوسان)

@dataclass
class Session:
    watches: List[Tuple[str,str]]    # (symbol, interval)
    params: Params
    filters: Filters
    ensemble: bool = False
    auto_task: Optional[asyncio.Task] = None
    auto_period: int = 60

SESSIONS: Dict[int, Session] = {}

# ---------- Utils ----------
def ensure_session(uid: int) -> Session:
    if uid not in SESSIONS:
        SESSIONS[uid] = Session(watches=[], params=Params(), filters=Filters())
    return SESSIONS[uid]

def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["time"] = pd.to_datetime(df["close_time"], unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df[["time","open","high","low","close","volume"]]

# ----- Strategy Implementations -----
def strat_ema_rsi(df: pd.DataFrame, p: Params) -> pd.Series:
    fast = ema(df["close"], p.ema_fast)
    slow = ema(df["close"], p.ema_slow)
    r = rsi(df["close"], p.rsi_len)
    buy = (fast > slow) & (r >= p.rsi_buy)
    sell = (fast < slow) & (r <= p.rsi_sell)
    return np.where(buy, 1, np.where(sell, -1, 0))

def strat_macd(df: pd.DataFrame, p: Params) -> pd.Series:
    macd = ema(df["close"], p.macd_fast) - ema(df["close"], p.macd_slow)
    sig = macd.ewm(span=p.macd_signal, adjust=False).mean()
    hist = macd - sig
    return np.where(hist > 0, 1, np.where(hist < 0, -1, 0))

def strat_donchian(df: pd.DataFrame, p: Params) -> pd.Series:
    hh = df["high"].rolling(p.donchian_len).max()
    ll = df["low"].rolling(p.donchian_len).min()
    mid = (hh + ll) / 2
    return np.where(df["close"] > mid, 1, np.where(df["close"] < mid, -1, 0))

def supertrend_series(df: pd.DataFrame, period: int, mult: float) -> pd.Series:
    a = atr(df, period)
    mid = (df["high"] + df["low"]) / 2
    upper = mid + mult * a
    lower = mid - mult * a
    st = pd.Series(index=df.index, dtype=float)
    dir_up = True
    st.iloc[0] = upper.iloc[0]
    for i in range(1, len(df)):
        prev = st.iloc[i-1]
        if dir_up:
            st.iloc[i] = min(upper.iloc[i], prev)
            if df["close"].iloc[i] < st.iloc[i]:
                dir_up = False
                st.iloc[i] = lower.iloc[i]
        else:
            st.iloc[i] = max(lower.iloc[i], prev)
            if df["close"].iloc[i] > st.iloc[i]:
                dir_up = True
                st.iloc[i] = upper.iloc[i]
    return st

def strat_supertrend(df: pd.DataFrame, p: Params) -> pd.Series:
    st = supertrend_series(df, p.st_period, p.st_mult)
    return np.where(df["close"] > st, 1, np.where(df["close"] < st, -1, 0))

def strat_bb_meanrev(df: pd.DataFrame, p: Params) -> pd.Series:
    ma = df["close"].rolling(p.bb_len).mean()
    std = df["close"].rolling(p.bb_len).std(ddof=0)
    upper, lower = ma + p.bb_k*std, ma - p.bb_k*std
    # mean-reversion: buy near lower band, sell near upper band
    sig = np.where(df["close"] < lower, 1, np.where(df["close"] > upper, -1, 0))
    return sig

STRATS = {
    "ema_rsi": strat_ema_rsi,
    "macd": strat_macd,
    "donchian": strat_donchian,
    "supertrend": strat_supertrend,
    "bb_meanrev": strat_bb_meanrev,
}

# ----- Filters & Score -----
def apply_filters(df: pd.DataFrame, sig: np.ndarray, f: Filters) -> Tuple[np.ndarray, np.ndarray]:
    """returns (filtered_signal, strength) where strength in {0,1,2}"""
    strength = np.zeros(len(sig), dtype=int)
    sig = sig.copy()
    mask = np.ones(len(sig), dtype=bool)
    if f.regime_ema > 0:
        reg = ema(df["close"], f.regime_ema)
        bull = df["close"] > reg
        bear = df["close"] < reg
        mask = mask & ((sig == 1) & bull | (sig == -1) & bear | (sig == 0))
    if f.vol_atr > 0 and f.min_atr_frac > 0:
        a = atr(df, f.vol_atr)
        frac = a / df["close"].replace(0, np.nan)
        mask = mask & (frac >= f.min_atr_frac)
    filtered = np.where(mask, sig, 0)
    # strength: 2 if passed both filters, 1 if passed one, 0 otherwise/no-trade
    passed_regime = (f.regime_ema == 0) or ((filtered != 0) & (ema(df["close"], f.regime_ema).notna()))
    passed_vol = (f.vol_atr == 0) or ((filtered != 0))
    strength = np.where(filtered == 0, 0, np.where(passed_regime & passed_vol, 2, 1))
    return filtered, strength

def ensemble_vote(df: pd.DataFrame, p: Params, f: Filters) -> Tuple[np.ndarray, np.ndarray]:
    votes = []
    weights = []
    for name in ["ema_rsi","macd","donchian","supertrend","bb_meanrev"]:
        s = STRATS[name](df, p)
        s, _ = apply_filters(df, s, f)
        votes.append(s)
        weights.append(1.0)
    votes = np.vstack(votes)
    w = np.array(weights)[:,None]
    score = np.sign((votes * w).sum(axis=0))
    # strength based on unanimity/magnitude
    mag = np.abs(votes).sum(axis=0)
    strength = np.where(score == 0, 0, np.where(mag >= 4, 2, 1))
    return score.astype(int), strength.astype(int)

# ----- Chart -----
def make_chart(df: pd.DataFrame, title: str, out_path: str):
    fig = plt.figure()
    plt.title(title)
    plt.plot(pd.to_datetime(df["time"]), df["close"])
    plt.xlabel("Time"); plt.ylabel("Price")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# ----- Telegram -----
def kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup([["/list","/add","/del","/strategy"],
                                ["/params","/filters","/ensemble on","/ensemble off"],
                                ["/run","/auto on 60","/auto off","/help"]],
                               resize_keyboard=True)

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ensure_session(update.effective_user.id)
    await update.message.reply_text(
        "سلام! نسخه PRO سیگنال‌دهی (آموزشی) فعال شد.\n/add BTCUSDT 1m  → جفت اضافه کن، /run برای سیگنال.",
        reply_markup=kb()
    )

async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "دستورها:\n"
        "/add SYMBOL INTERVAL — مثال: /add BTCUSDT 1m\n"
        "/del SYMBOL — حذف از واچ‌لیست\n"
        "/list — نمایش واچ‌لیست\n"
        "/strategy ema_rsi|macd|donchian|supertrend|bb_meanrev\n"
        "/params key=val ...  (مثال: /params ema_fast=12 ema_slow=26 rsi_len=14)\n"
        "/filters key=val ...  (مثال: /filters regime_ema=200 vol_atr=14 min_atr_frac=0.001)\n"
        "/ensemble on|off — رای‌گیری بین چند استراتژی\n"
        "/run — سیگنال آخر + تصویر\n"
        "/auto on N — اجرا هر N ثانیه | /auto off\n"
        "⚠️ آموزشی؛ سود تضمینی وجود ندارد."
    )

async def add_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = ensure_session(update.effective_user.id)
    if len(ctx.args) < 2:
        await update.message.reply_text("فرمت: /add SYMBOL INTERVAL")
        return
    sym, interval = ctx.args[0].upper(), ctx.args[1]
    if interval not in VALID_INTERVALS:
        await update.message.reply_text("Interval نامعتبر.")
        return
    if (sym, interval) in s.watches:
        await update.message.reply_text("قبلاً اضافه شده.")
        return
    s.watches.append((sym, interval))
    await update.message.reply_text(f"✅ اضافه شد: {sym} {interval}")

async def del_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = ensure_session(update.effective_user.id)
    if not ctx.args:
        await update.message.reply_text("فرمت: /del SYMBOL")
        return
    sym = ctx.args[0].upper()
    before = len(s.watches)
    s.watches = [(sy,iv) for sy,iv in s.watches if sy != sym]
    await update.message.reply_text("حذف شد." if len(s.watches) < before else "چیزی پیدا نشد.")

async def list_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = ensure_session(update.effective_user.id)
    if not s.watches:
        await update.message.reply_text("لیست خالی است.")
        return
    await update.message.reply_text("واچ‌لیست:\n" + "\n".join(f"- {sy} {iv}" for sy,iv in s.watches))

async def strategy_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = ensure_session(update.effective_user.id)
    if ctx.args and ctx.args[0] in STRATS:
        s.params.strategy = ctx.args[0]
        await update.message.reply_text(f"✅ استراتژی: {s.params.strategy}")
    else:
        await update.message.reply_text("انتخاب کن: " + " | ".join(STRATS.keys()))

def _apply_params_obj(obj, kv: Dict[str,str]):
    updated = {}
    for k, v in kv.items():
        if not hasattr(obj, k):
            continue
        cur = getattr(obj, k)
        try:
            if isinstance(cur, int):
                val = int(float(v))
            elif isinstance(cur, float):
                val = float(v)
            else:
                val = str(v)
            setattr(obj, k, val); updated[k] = val
        except Exception:
            pass
    return updated

async def params_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = ensure_session(update.effective_user.id)
    kv = {}
    for arg in ctx.args:
        if "=" in arg:
            k,v = arg.split("=",1)
            kv[k.strip()] = v.strip()
    updated = _apply_params_obj(s.params, kv)
    msg = ("✅ بروزرسانی شد.\n" if updated else "") + \
          "پارامترها:\n" + "\n".join(f"{k}={v}" for k,v in asdict(s.params).items())
    await update.message.reply_text(msg)

async def filters_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = ensure_session(update.effective_user.id)
    kv = {}
    for arg in ctx.args:
        if "=" in arg:
            k,v = arg.split("=",1)
            kv[k.strip()] = v.strip()
    updated = _apply_params_obj(s.filters, kv)
    msg = ("✅ بروزرسانی شد.\n" if updated else "") + \
          "فیلترها:\n" + "\n".join(f"{k}={v}" for k,v in asdict(s.filters).items())
    await update.message.reply_text(msg)

async def ensemble_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = ensure_session(update.effective_user.id)
    if ctx.args and ctx.args[0].lower() == "on":
        s.ensemble = True
        await update.message.reply_text("✅ Ensemble روشن شد.")
    else:
        s.ensemble = False
        await update.message.reply_text("⏹ Ensemble خاموش شد.")

def fetch_and_signal(symbol: str, interval: str, s: Session) -> Tuple[str, float, int]:
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": s.params.lookback}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["time"] = pd.to_datetime(df["close_time"], unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df = df[["time","open","high","low","close","volume"]]
    # signal
    if s.ensemble:
        sig, strength = ensemble_vote(df, s.params, s.filters)
    else:
        base = STRATS[s.params.strategy](df, s.params)
        base, strength = apply_filters(df, base, s.filters)
        sig = base
    label = int(sig[-1]) if len(sig) else 0
    price = float(df["close"].iloc[-1])
    power = int(strength[-1]) if len(sig) else 0
    # chart
    os.makedirs("out", exist_ok=True)
    chart_path = f"out/{symbol}_{interval}.png"
    make_chart(df.tail(200), f"{symbol} {interval} — {s.params.strategy.upper()}{' [ENS]' if s.ensemble else ''}", chart_path)
    return chart_path, price, label if label in (-1,0,1) else 0, power

async def run_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = ensure_session(update.effective_user.id)
    if not s.watches:
        await update.message.reply_text("با /add اضافه کن. مثال: /add BTCUSDT 5m")
        return
    for sym, iv in s.watches:
        try:
            chart, price, label, power = fetch_and_signal(sym, iv, s)
            sigtxt = "BUY" if label==1 else "SELL" if label==-1 else "NO_TRADE"
            strength = "قوی" if power==2 else "ضعیف" if power==1 else "—"
            cap = f"📈 {sym} {iv}\nسیگنال: {sigtxt} ({strength})\nقیمت: {price:.6f}"
            await update.message.reply_photo(photo=open(chart, "rb"), caption=cap)
            await asyncio.sleep(0.4)
        except Exception as e:
            log.exception("run error")
            await update.message.reply_text(f"خطا برای {sym} {iv}: {e}")

async def auto_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = ensure_session(update.effective_user.id)
    mode = ctx.args[0].lower() if ctx.args else "on"
    if mode == "on":
        period = int(ctx.args[1]) if len(ctx.args) > 1 else s.auto_period
        s.auto_period = max(20, period)
        if s.auto_task and not s.auto_task.done():
            await update.message.reply_text("اتوران فعال است.")
            return
        async def loop():
            while True:
                try:
                    await run_cmd(update, ctx)
                except Exception:
                    pass
                await asyncio.sleep(s.auto_period)
        s.auto_task = asyncio.create_task(loop())
        await update.message.reply_text(f"✅ اتوران روشن شد ({s.auto_period}s). /auto off برای توقف.")
    else:
        if s.auto_task:
            s.auto_task.cancel()
            s.auto_task = None
        await update.message.reply_text("⏹ اتوران خاموش شد.")

async def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN env var not set")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("add", add_cmd))
    app.add_handler(CommandHandler("del", del_cmd))
    app.add_handler(CommandHandler("list", list_cmd))
    app.add_handler(CommandHandler("strategy", strategy_cmd))
    app.add_handler(CommandHandler("params", params_cmd))
    app.add_handler(CommandHandler("filters", filters_cmd))
    app.add_handler(CommandHandler("ensemble", ensemble_cmd))
    app.add_handler(CommandHandler("run", run_cmd))
    app.add_handler(CommandHandler("auto", auto_cmd))

    log.info("Signal bot PRO running...")
    await app.run_polling()

if __name__ == "__main__":
    from telegram.ext import Application, CommandHandler, ContextTypes
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
