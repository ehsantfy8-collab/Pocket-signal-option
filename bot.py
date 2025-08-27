import telebot

# 🔑 اطلاعات ربات
TOKEN = "8021948001:AAFtM1XlwyYS3Xx33GJaZLM8g56zvI4murc"
CHAT_ID = "415392967"

bot = telebot.TeleBot(TOKEN)

# وقتی /start بزنی
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.chat.id, "ربات با موفقیت فعال شد ✅")

# تست ساده برای ارسال سیگنال
@bot.message_handler(commands=['signal'])
def send_signal(message):
    bot.send_message(CHAT_ID, "📢 سیگنال تستی ارسال شد!")

# اجرا
print("ربات روشن شد ✅")
bot.polling()
