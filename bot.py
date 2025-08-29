import os
import time
import threading
import requests
from bs4 import BeautifulSoup
from telegram import Bot
from flask import Flask

# گرفتن توکن و چت آی‌دی از Environment Variable
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

bot = Bot(token=TELEGRAM_TOKEN)
app = Flask(__name__)

URL = "https://www.investing.com/forex-signals"

def get_signals():
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(URL, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    signals = []
    rows = soup.select("table tr")

    for row in rows[1:6]:  # فقط ۵ سیگنال اول
        cols = row.find_all("td")
        if len(cols) >= 5:
            pair = cols[0].get_text(strip=True)
            action = cols[1].get_text(strip=True)
            timeframe = cols[2].get_text(strip=True)
            strength = cols[3].get_text(strip=True)

            signals.append(f"📊 {pair} | {timeframe}\n📈 {action} ({strength})")

    return signals

def send_signals():
    sent = set()
    while True:
        try:
            signals = get_signals()
            for s in signals:
                if s not in sent:
                    bot.send_message(chat_id=CHAT_ID, text=s)
                    sent.add(s)
            time.sleep(300)  # هر ۵ دقیقه
        except Exception as e:
            bot.send_message(chat_id=CHAT_ID, text=f"⚠️ Error: {e}")
            time.sleep(60)

@app.route("/")
def home():
    return "📡 Forex Signal Bot is running on Render 🚀"

if __name__ == "__main__":
    t = threading.Thread(target=send_signals)
    t.start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
