import os
import time
import threading
import requests
from bs4 import BeautifulSoup
from telegram import Bot
from flask import Flask

# گرفتن توکن و چت آی‌دی از Environment
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

bot = Bot(token=TELEGRAM_TOKEN)
URL = "https://www.investing.com/forex-signals"

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Bot is running!"

def get_signals():
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(URL, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    signals = []
    rows = soup.select("table tr")
    for row in rows[1:6]:
        cols = row.find_all("td")
        if len(cols) >= 5:
            pair = cols[0].get_text(strip=True)
            action = cols[1].get_text(strip=True)
            timeframe = cols[2].get_text(strip=True)
            strength = cols[3].get_text(strip=True)
            signals.append(f"📊 {pair} | {timeframe}\n📈 {action} ({strength})")
    return signals

def run_bot():
    sent = set()
    while True:
        try:
            signals = get_signals()
            for s in signals:
                if s not in sent:
                    bot.send_message(chat_id=CHAT_ID, text=s)
                    sent.add(s)
            time.sleep(300)
        except Exception as e:
            bot.send_message(chat_id=CHAT_ID, text=f"⚠️ Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    # اجرای ربات توی Thread جدا
    threading.Thread(target=run_bot, daemon=True).start()
    # اجرای Flask روی پورت Render
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
