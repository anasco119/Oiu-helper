import os
import re
import sqlite3
from aiogram import Bot, Dispatcher, types
from aiogram.types import ChatPermissions
from aiogram.utils import executor

BOT_TOKEN = os.getenv("BOT_TOKEN")
GROUP_ID = int(os.getenv("GROUP_ID"))
ADMIN_IDS = list(map(int, os.getenv("ADMIN_IDS", "").split(",")))

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

conn = sqlite3.connect("warnings.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS warnings (
    user_id INTEGER PRIMARY KEY,
    count INTEGER
)
""")
conn.commit()

LINK_PATTERN = re.compile(r"(http|https|t\.me|bit\.ly|\.com|\.net|\.org)")
MENTION_PATTERN = re.compile(r"@\w+")

MAX_WARNINGS = 2

async def restrict_user(chat_id: int, user_id: int, can_send_messages: bool):
    permissions = ChatPermissions(can_send_messages=can_send_messages)
    await bot.restrict_chat_member(chat_id, user_id, permissions=permissions)

@dp.message_handler(content_types=types.ContentTypes.TEXT, chat_type=types.ChatType.GROUP)
async def monitor_messages(message: types.Message):
    if message.chat.id != GROUP_ID:
        return

    user_id = message.from_user.id
    if user_id in ADMIN_IDS:
        return

    text = message.text or ""

    is_spam = False
    reasons = []

    if LINK_PATTERN.search(text):
        is_spam = True
        reasons.append("روابط خارجية")

    if MENTION_PATTERN.search(text):
        is_spam = True
        reasons.append("إشارات غير مصرّح بها")

    if is_spam:
        try:
            await message.delete()
        except:
            pass

        cursor.execute("SELECT count FROM warnings WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()

        if row is None:
            cursor.execute("INSERT INTO warnings (user_id, count) VALUES (?, ?)", (user_id, 1))
            conn.commit()
            await restrict_user(message.chat.id, user_id, False)
            await message.reply(f"🚫 @{message.from_user.username or message.from_user.first_name} تم كتمك بسبب: {', '.join(reasons)}. (تحذير أول)")
        elif row[0] < MAX_WARNINGS - 1:
            cursor.execute("UPDATE warnings SET count = count + 1 WHERE user_id = ?", (user_id,))
            conn.commit()
            await restrict_user(message.chat.id, user_id, False)
            await message.reply(f"⚠️ @{message.from_user.username or message.from_user.first_name} هذا هو التحذير الثاني. تكرار ذلك سيؤدي للحظر.")
        else:
            await message.reply(f"❌ @{message.from_user.username or message.from_user.first_name} تم حظرك بسبب تكرار المخالفات.")
            await bot.ban_chat_member(message.chat.id, user_id)
            cursor.execute("DELETE FROM warnings WHERE user_id = ?", (user_id,))
            conn.commit()

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
