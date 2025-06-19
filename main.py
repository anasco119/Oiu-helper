import os
import re
import sqlite3
from datetime import datetime, timedelta
from flask import Flask
import telebot
from telebot.types import ChatPermissions

# متغيرات البيئة
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROUP_ID = int(os.getenv("GROUP_ID"))
ADMIN_IDS = list(map(int, os.getenv("ADMIN_IDS", "").split(",")))
MAIN_ADMIN_ID = int(os.getenv("MAIN_ADMIN_ID"))

bot = telebot.TeleBot(BOT_TOKEN)

# قاعدة البيانات
conn = sqlite3.connect("warnings.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS warnings (
    user_id INTEGER PRIMARY KEY,
    count INTEGER
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS reported_groups (
    group_id INTEGER PRIMARY KEY
)
""")
conn.commit()

# أنماط
LINK_PATTERN = re.compile(r"(http|https|t\.me|bit\.ly|\.com|\.net|\.org)")
MENTION_PATTERN = re.compile(r"@\w+")
MAX_WARNINGS = 2

def notify_new_group(message):
    group_id = message.chat.id
    group_title = message.chat.title or "بدون اسم"
    cursor.execute("SELECT group_id FROM reported_groups WHERE group_id = ?", (group_id,))
    if cursor.fetchone() is None:
        try:
            bot.send_message(
                MAIN_ADMIN_ID,
                f"📌 تم إضافة البوت إلى مجموعة جديدة:\n📍 الاسم: {group_title}\n🆔 ID: `{group_id}`",
                parse_mode="Markdown"
            )
            cursor.execute("INSERT INTO reported_groups (group_id) VALUES (?)", (group_id,))
            conn.commit()
        except Exception as e:
            print(f"⚠️ فشل إرسال إشعار للمشرف: {e}")

def is_spam(text, mentioned_usernames, group_members_usernames):
    reasons = []
    if LINK_PATTERN.search(text):
        reasons.append("روابط")
    for mention in mentioned_usernames:
        if mention not in group_members_usernames:
            reasons.append("إشارات خارجية")
            break
    return reasons

@bot.message_handler(func=lambda message: True, content_types=['text'])
def handle_message(message):
    if message.chat.type != "supergroup":
        return

    notify_new_group(message)

    if message.chat.id != GROUP_ID:
        return

    user_id = message.from_user.id
    if user_id in ADMIN_IDS:
        return

    text = message.text or ""
    mentioned_usernames = re.findall(r"@(\w+)", text)

    try:
        admins = bot.get_chat_administrators(GROUP_ID)
        members = [admin.user.username for admin in admins if admin.user.username]
    except:
        members = []

    reasons = is_spam(text, mentioned_usernames, members)
    if not reasons:
        return

    try:
        bot.delete_message(message.chat.id, message.message_id)
    except:
        pass

    cursor.execute("SELECT count FROM warnings WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    until = datetime.utcnow() + timedelta(seconds=10)

    if row is None:
        cursor.execute("INSERT INTO warnings (user_id, count) VALUES (?, ?)", (user_id, 1))
        conn.commit()
        try:
            bot.restrict_chat_member(
                message.chat.id,
                user_id,
                ChatPermissions(can_send_messages=False),
                until_date=until
            )
        except:
            pass
        bot.send_message(
            message.chat.id,
            f"🚫 @{message.from_user.username or message.from_user.first_name} تم كتمك لمدة 10 ثوانٍ بسبب: {', '.join(reasons)}. (تحذير أول)"
        )
    elif row[0] < MAX_WARNINGS - 1:
        cursor.execute("UPDATE warnings SET count = count + 1 WHERE user_id = ?", (user_id,))
        conn.commit()
        try:
            bot.restrict_chat_member(
                message.chat.id,
                user_id,
                ChatPermissions(can_send_messages=False),
                until_date=until
            )
        except:
            pass
        bot.send_message(
            message.chat.id,
            f"⚠️ @{message.from_user.username or message.from_user.first_name} تم كتمك لمدة 10 ثوانٍ (تحذير ثاني). التكرار سيؤدي إلى الحظر."
        )
    else:
        try:
            bot.ban_chat_member(message.chat.id, user_id)
        except:
            pass
        bot.send_message(
            message.chat.id,
            f"❌ @{message.from_user.username or message.from_user.first_name} تم حظرك بسبب تكرار المخالفات."
        )
        cursor.execute("DELETE FROM warnings WHERE user_id = ?", (user_id,))
        conn.commit()

# واجهة Flask للفحص
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ البوت يعمل الآن"

# بدء البوت
import threading

# تشغيل بوت تيليغرام في Thread منفصل
def run_bot():
    print("🤖 Bot polling started...")
    bot.infinity_polling()

threading.Thread(target=run_bot).start()

# تشغيل خادم Flask على المنفذ الذي تحدده Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render يوفر PORT كمتغير بيئة
    app.run(host="0.0.0.0", port=port)
