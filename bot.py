import os
import requests
import asyncio
from bs4 import BeautifulSoup
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# گرفتن توکن و چت‌آیدی از Environment Variable
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

URL = "https://www.investing.com/forex-signals"

def get_signals():
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(URL, headers=headers, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    signals = []
    rows = soup.select("table tr")

    for row in rows[1:6]:  # فقط ۵ سیگنال اول
        cols = row.find_all("td")
        if len(cols) >= 4:
            pair = cols[0].get_text(strip=True)
            action = cols[1].get_text(strip=True)
            timeframe = cols[2].get_text(strip=True)
            strength = cols[3].get_text(strip=True)
            signals.append(f"📊 {pair} | {timeframe}\n📈 {action} ({strength})")

    return signals


# --- دستورات بات ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("سلام 👋 من بات سیگنال فارکس هستم.\nاز /signals استفاده کن برای دیدن آخرین سیگنال‌ها.")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📌 دستورات:\n/start - شروع\n/signals - آخرین سیگنال‌ها\n/help - راهنما")

async def signals_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        signals = get_signals()
        if signals:
            for s in signals:
                await update.message.reply_text(s)
        else:
            await update.message.reply_text("هیچ سیگنالی پیدا نشد ❌")
    except Exception as e:
        await update.message.reply_text(f"⚠️ خطا: {e}")


# --- ارسال خودکار هر ۵ دقیقه ---
async def auto_send_signals(app: Application):
    sent = set()
    while True:
        try:
            signals = get_signals()
            for s in signals:
                if s not in sent:
                    await app.bot.send_message(chat_id=CHAT_ID, text=s)
                    sent.add(s)
        except Exception as e:
            await app.bot.send_message(chat_id=CHAT_ID, text=f"⚠️ Error: {e}")
        await asyncio.sleep(300)  # هر ۵ دقیقه


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # دستورات
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("signals", signals_cmd))

    # اجرای همزمان: بات + ارسال اتوماتیک
    app.job_queue.run_repeating(lambda ctx: asyncio.create_task(auto_send_signals(app)), interval=300, first=1)

    print("🤖 Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
