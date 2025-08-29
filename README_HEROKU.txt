# Deploy Telegram Signal Bot PRO on Heroku

Files in this package:
- signal_bot_pro.py
- requirements.txt
- Procfile
- runtime.txt

## 1) Create the app
heroku login
heroku create my-signal-bot

## 2) Set your Telegram token
heroku config:set TELEGRAM_BOT_TOKEN=123456:ABC-yourToken

## 3) Deploy
git init
git add .
git commit -m "deploy bot"
git push heroku master

## 4) Logs
heroku logs --tail

Notes:
- This bot sends **signals only** (educational). It does NOT connect to brokers.
- Uses python-telegram-bot v21.
