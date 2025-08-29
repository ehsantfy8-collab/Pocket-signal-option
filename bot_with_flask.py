from flask import Flask
import telebot
import os

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # توی Render باید ست کنی
bot = telebot.TeleBot(TOKEN)

app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is running on Render!"

# یک دستور ساده تست
@bot.message_handler(commands=["start"])
def start_message(message):
    bot.reply_to(message, "سلام 🌹 ربات روشنه و داره روی Render کار می‌کنه!")

# برای اینکه بات به صورت polling کار کنه
import threading

def run_polling():
    bot.infinity_polling()

threading.Thread(target=run_polling).start()
