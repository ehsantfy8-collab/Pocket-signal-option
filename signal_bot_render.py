import asyncio
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update
import os

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # از متغیر محیطی در Render می‌گیره

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ ربات آنلاین شد و روی Render اجرا میشه!")

def main():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))

    print("🚀 Bot is running on Render...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    asyncio.run(main())
