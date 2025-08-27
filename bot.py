import telebot

TOKEN = "8021948001:AAFtM1XlwyYS3Xx33GJaZLM8g56zvI4murc"
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(message, "سلام! ربات شما روشن است.")

bot.polling()
