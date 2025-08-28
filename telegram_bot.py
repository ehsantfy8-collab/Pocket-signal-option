
import os, requests
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
def send_message(text: str):
    if not (BOT_TOKEN and CHAT_ID):
        print("[Telegram] Missing BOT_TOKEN/CHAT_ID, printing message instead:\n", text)
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print("[Telegram] Error:", r.text)
    except Exception as e:
        print("[Telegram] Exception:", e)
