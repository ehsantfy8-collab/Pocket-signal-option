import os
import telebot
from flask import Flask, request

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = telebot.TeleBot(TOKEN)
app = Flask(__name__)

CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # اختیاری

@app.route('/')
def home():
    return "Bot is running!"

@app.route(f'/{TOKEN}', methods=['POST'])
def webhook():
    json_str = request.stream.read().decode("UTF-8")
    update = telebot.types.Update.de_json(json_str)
    bot.process_new_updates([update])
    return "OK", 200

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "رباتت روی Render فعاله ✅")

@bot.message_handler(func=lambda m: True)
def echo_all(message):
    bot.reply_to(message, f"پیام گرفتم: {message.text}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    bot.remove_webhook()
    bot.set_webhook(url=f"{os.environ.get('RENDER_EXTERNAL_URL')}/{TOKEN}")
    app.run(host="0.0.0.0", port=port)
