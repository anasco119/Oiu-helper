import os
import re
import sqlite3
from telegram import Update, ChatPermissions
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# تحميل متغيرات البيئة
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROUP_ID = int(os.getenv("GROUP_ID"))
ADMIN_IDS = list(map(int, os.getenv("ADMIN_IDS", "").split(",")))  # مفصولة بفواصل

# قاعدة البيانات
conn = sqlite3.connect("warnings.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS warnings (
    user_id INTEGER PRIMARY KEY,
    count INTEGER
)
""")
conn.commit()

# أنماط الكشف
LINK_PATTERN = re.compile(r"(http|https|t\.me|bit\.ly|\.com|\.net|\.org)")
MENTION_PATTERN = re.compile(r"@\w+")

MAX_WARNINGS = 2  # عدد التحذيرات قبل الحظر

async def monitor_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    message = update.message
    user = message.from_user
    chat = update.effective_chat
    user_id = user.id
    text = message.text.strip()

    if chat.id != GROUP_ID:
        return  # نتجاهل الرسائل في غير المجموعة المستهدفة

    if user_id in ADMIN_IDS:
        return  # لا نتخذ إجراء ضد المشرفين في القائمة البيضاء

    is_spam = False
    reasons = []

    # كشف الروابط
    if LINK_PATTERN.search(text):
        is_spam = True
        reasons.append("روابط مشبوهة")

    # كشف الإشارات الخارجية
    if MENTION_PATTERN.search(text):
        is_spam = True
        reasons.append("إشارات لأعضاء خارج المجموعة")

    if is_spam:
        try:
            await message.delete()
        except:
            pass

        # عدد التحذيرات الحالي
        cursor.execute("SELECT count FROM warnings WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()

        if row is None:
            # أول تحذير
            cursor.execute("INSERT INTO warnings (user_id, count) VALUES (?, ?)", (user_id, 1))
            conn.commit()
            await context.bot.restrict_chat_member(chat.id, user_id, ChatPermissions(can_send_messages=False))
            await chat.send_message(
                f"🚫 @{user.username or user.first_name} تم كتمك بسبب: {', '.join(reasons)}.\n(التحذير الأول)"
            )
        elif row[0] < MAX_WARNINGS - 1:
            # ثاني تحذير
            cursor.execute("UPDATE warnings SET count = count + 1 WHERE user_id = ?", (user_id,))
            conn.commit()
            await context.bot.restrict_chat_member(chat.id, user_id, ChatPermissions(can_send_messages=False))
            await chat.send_message(
                f"⚠️ @{user.username or user.first_name} هذا هو التحذير الثاني.\nفي حال التكرار سيتم حظرك."
            )
        else:
            # حظر المستخدم
            await chat.send_message(
                f"❌ @{user.username or user.first_name} تم حظرك بسبب تكرار المخالفات."
            )
            await context.bot.ban_chat_member(chat.id, user_id)
            cursor.execute("DELETE FROM warnings WHERE user_id = ?", (user_id,))
            conn.commit()

if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & filters.ChatType.GROUPS, monitor_messages))

    print("🤖 Bot is running...")
    app.run_polling()
