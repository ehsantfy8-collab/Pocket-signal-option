
from flask import Flask, request, jsonify
import os
from telegram_bot import send_message
from strategy import analyze_symbol
from data import fetch_candles

app = Flask(__name__)

@app.route("/health")
def health():
    return "ok", 200

@app.route("/analyze", methods=["POST"])
def analyze():
    body = request.get_json(force=True) or {}
    symbol = body.get("symbol", "EURUSD=X")
    tf = body.get("timeframe", "1m")
    lookback = int(body.get("lookback", 300))
    candles = fetch_candles(symbol, tf, lookback)
    result = analyze_symbol(symbol, candles, tf)
    return jsonify(result), 200

@app.route("/tradeview", methods=["POST"])
def tradeview():
    raw = request.data.decode("utf-8", errors="ignore")
    payload = request.get_json(silent=True) or {}
    symbol = payload.get("symbol")
    timeframe = payload.get("timeframe", "1m")
    if symbol:
        try:
            candles = fetch_candles(symbol, timeframe, 300)
            result = analyze_symbol(symbol, candles, timeframe)
            msg = f"📣 TradingView Alert\n{raw}\n\n📊 تحلیل خودکار:\n{result.get('text')}"
        except Exception as e:
            msg = f"📣 TradingView Alert\n{raw}\n\n⚠️ تحلیل خودکار ناموفق: {e}"
    else:
        msg = f"📣 TradingView Alert\n{raw}"
    send_message(msg)
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
