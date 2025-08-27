import telebot

# توکن و چت آیدی خودت
TOKEN = "8021948001:AAFtM1XlwyYS3Xx33GJaZLM8g56zvI4murc"
CHAT_ID = "415392967"

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.chat.id, "✅ ربات با موفقیت فعال شد!")

@bot.message_handler(commands=['signal'])
def send_signal(message):
    bot.send_message(CHAT_ID, "📢 سیگنال تستی ارسال شد!")

print("ربات روشن شد ✅")
bot.polling()
