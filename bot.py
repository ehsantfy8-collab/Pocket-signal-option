import os
import time
import requests
from bs4 import BeautifulSoup
from telegram import Bot

# گرفتن توکن و چت آی‌دی از Environment Variable
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

bot = Bot(token=TELEGRAM_TOKEN)

URL = "https://www.investing.com/forex-signals"

def get_signals():
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    r = requests.get(URL, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    signals = []
    rows = soup.select("table tr")

    for row in rows[1:6]:  # فقط 5 سیگنال اول
        cols = row.find_all("td")
        if len(cols) >= 5:
            pair = cols[0].get_text(strip=True)
            action = cols[1].get_text(strip=True)
            timeframe = cols[2].get_text(strip=True)
            strength = cols[3].get_text(strip=True)

            signals.append(f"📊 {pair} | {timeframe}\n📈 {action} ({strength})")

    return signals

def main():
    sent = set()
    while True:
        try:
            signals = get_signals()
            for s in signals:
                if s not in sent:
                    bot.send_message(chat_id=CHAT_ID, text=s)
                    sent.add(s)
            time.sleep(300)  # هر 5 دقیقه
        except Exception as e:
            bot.send_message(chat_id=CHAT_ID, text=f"⚠️ Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
