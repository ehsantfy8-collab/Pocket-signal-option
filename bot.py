
import telebot

TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.reply_to(message, "سلام 👋 ربات شما روشن است و آماده دریافت دستور است.")

@bot.message_handler(commands=['signal'])
def signal_message(message):
    bot.reply_to(message, "📊 سیگنال آزمایشی ارسال شد! (این فقط تست است)")

print("🤖 Bot is running...")
bot.infinity_polling()
