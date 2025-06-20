import os
import re
import sqlite3
from datetime import datetime, timedelta
from flask import Flask
import telebot
from telebot.types import ChatPermissions
import threading
import logging
from dotenv import load_dotenv
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import docx
import fitz                     # PyMuPDF
import google.generativeai as genai

# متغيرات البيئة
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROUP_ID = int(os.getenv("GROUP_ID"))
ADMIN_ID = int(os.getenv("ADMIN_ID"))
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

bot = telebot.TeleBot(BOT_TOKEN)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# تهيئة مكتبة Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    logging.info("Gemini configured successfully")
except Exception as e:
    logging.error(f"Error configuring Gemini: {e}")


# إنشاء نموذج GenerativeModel
model = genai.GenerativeModel('gemini-2.0-flash')

def generate_gemini_response(prompt):
    try:
        response = model.generate_content(prompt)
        if response.text:
            return response.text
        else:
            logging.error("No response text from Gemini.")
            return "No response from Gemini."
    except Exception as e:
        logging.error(f"Error in generate_gemini_response: {e}")
        return f"Error: {str(e)}"

# -------------------------------------------------------------------
#         Helper to call your Gemini/OpenRouter API
# -------------------------------------------------------------------

#     ...

# -------------------------------------------------------------------
#                       Configuration
# -------------------------------------------------------------------


# -------------------------------------------------------------------
#                  Logging & Database Setup
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

conn   = sqlite3.connect("quiz_users.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id     INTEGER PRIMARY KEY,
    major       TEXT,
    quiz_count  INTEGER DEFAULT 0,
    last_reset  TEXT
)
""")
conn.commit()

# track temporary state for custom-major input
user_states = {}

# -------------------------------------------------------------------
#                     Text Extraction & OCR
# -------------------------------------------------------------------
def extract_text_from_pdf(path: str) -> str:
    try:
        doc = fitz.open(path)
        text = "\n".join([page.get_text() for page in doc])
        return text.strip()
    except Exception as e:
        logging.error(f"Error extracting PDF text: {e}")
        return ""
    # fallback to PyMuPDF text extraction
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

# -------------------------------------------------------------------
#                     Quota Management
# -------------------------------------------------------------------
def reset_if_needed(user_id: int):
    this_month = datetime.now().strftime("%Y-%m")
    cursor.execute("SELECT last_reset FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    if not row or row[0] != this_month:
        cursor.execute("""
            INSERT OR REPLACE INTO users(user_id, major, quiz_count, last_reset)
            VALUES (?, COALESCE((SELECT major FROM users WHERE user_id=?), ''), 0, ?)
        """, (user_id, user_id, this_month))
        conn.commit()

def can_generate(user_id: int) -> bool:
    reset_if_needed(user_id)
    cursor.execute("SELECT quiz_count FROM users WHERE user_id = ?", (user_id,))
    count = cursor.fetchone()[0]
    return count < 3

def increment_count(user_id: int):
    cursor.execute("UPDATE users SET quiz_count = quiz_count + 1 WHERE user_id = ?", (user_id,))
    conn.commit()

# -------------------------------------------------------------------
#                 Quiz Generation & Formatting
# -------------------------------------------------------------------
def generate_quizzes_from_text(text: str, major: str, num_quizzes: int = 10):
    prompt = (
        f"Create {num_quizzes} quizzes for this student "
        f"based on the following content. The questions should be tailored to {major} students. "
        "Each quiz should have 4 options and one correct answer. The quizzes should vary in style: "
        "fill-in-the-blank, multiple-choice, and sentence completion and with the same language of the content.\n\n"
        "Format:\n"
        "Question: ...\n"
        "Option 1: ...\n"
        "Option 2: ...\n"
        "Option 3: ...\n"
        "Option 4: ...\n"
        "Correct Answer: ...\n\n"
        f"Content:\n{text}"
    )
    raw = generate_gemini_response(prompt)
    # extract via regex
    pattern = (
        r"Question:\s*(.*?)\n"
        r"Option 1:\s*(.*?)\n"
        r"Option 2:\s*(.*?)\n"
        r"Option 3:\s*(.*?)\n"
        r"Option 4:\s*(.*?)\n"
        r"Correct Answer:\s*(.*?)(?=\n\n|$)"
    )
    matches = re.findall(pattern, raw, re.DOTALL)
    quizzes = []
    for q, a1, a2, a3, a4, corr in matches:
        opts = [a1.strip(), a2.strip(), a3.strip(), a4.strip()]
        try:
            corr_idx = opts.index(corr.strip())
        except ValueError:
            continue
        quizzes.append((q.strip(), opts, corr_idx))
    return quizzes

def send_quizzes_as_polls(chat_id: int, quizzes: list):
    for q, opts, corr_idx in quizzes:
        bot.send_poll(
            chat_id=chat_id,
            question=q,
            options=opts,
            type="quiz",
            correct_option_id=corr_idx,
            is_anonymous=True
        )
    # final info message
    bot.send_message(chat_id,
        "🧪 📌 قريبًا: ميزة إنشاء اختبار تفاعلي تلقائي من هذه الأسئلة! 💡"
    )

# -------------------------------------------------------------------
#                  Telegram Bot Handlers
# -------------------------------------------------------------------
@bot.message_handler(commands=['start'])
def cmd_start(msg):
    keyboard = InlineKeyboardMarkup()
    buttons = [
        ("🩺 الطب", "major_الطب"),
        ("🛠️ الهندسة", "major_الهندسة"),
        ("💊 الصيدلة", "major_الصيدلة"),
        ("🗣️ اللغات", "major_اللغات"),
        ("❓ غير ذلك...", "major_custom"),
    ]
    for text, data in buttons:
        keyboard.add(InlineKeyboardButton(text, callback_data=data))

    bot.send_message(
        msg.chat.id,
        "👋 مرحبًا بك في البوت التعليمي الذكي!\n\n"
        "🎯 هذا البوت يساعدك على توليد اختبارات ذكية من ملفاتك الدراسية أو النصوص، حسب تخصصك.\n"
        "📌 متاح لك 3 اختبارات مجانية شهريًا.\n"
        "اختر تخصصك للبدء 👇",
        reply_markup=keyboard
    )

@bot.callback_query_handler(func=lambda c: c.data.startswith("major_"))
def cb_major(c):
    sel = c.data.split("_", 1)[1]
    uid = c.from_user.id

    if sel == "custom":
        user_states[uid] = "awaiting_major"
        bot.send_message(uid, "✏️ من فضلك أرسل اسم تخصصك بدقة.")
    else:
        # set directly
        cursor.execute("INSERT OR REPLACE INTO users(user_id, major) VALUES(?, ?)", (uid, sel))
        conn.commit()
        bot.send_message(uid,
            f"✅ تم تحديد تخصصك: {sel}\n"
            "الآن أرسل ملف (PDF/DOCX/TXT) أو نصًا مباشرًا لتوليد اختبارك."
        )

@bot.message_handler(func=lambda m: user_states.get(m.from_user.id) == "awaiting_major", content_types=['text'])
def set_custom_major(msg):
    major = msg.text.strip()
    uid   = msg.from_user.id

    cursor.execute(
        "INSERT OR REPLACE INTO users(user_id, major) VALUES(?, ?)",
        (uid, major)
    )
    conn.commit()
    user_states.pop(uid, None)

    bot.send_message(uid,
        f"✅ تم تسجيل تخصصك: \"{major}\"\n"
        "الآن أرسل ملف (PDF/DOCX/TXT) أو نصًا مباشرًا لتوليد اختبارك."
    )
    # notify admin
    bot.send_message(ADMIN_ID,
        f"🆕 تخصص جديد أُرسل من المستخدم:\n"
        f"👤 @{msg.from_user.username or msg.from_user.id}\n"
        f"📚 التخصص: {major}"
    )

@bot.message_handler(content_types=['document'])
def handle_document(msg):
    uid = msg.chat.id
    if not can_generate(uid):
        return bot.send_message(uid, "⚠️ لقد استنفدت 3 اختبارات مجانية هذا الشهر.")

    file_info = bot.get_file(msg.document.file_id)
    data      = bot.download_file(file_info.file_path)
    os.makedirs("downloads", exist_ok=True)
    path = os.path.join("downloads", msg.document.file_name)
    with open(path, "wb") as f:
        f.write(data)

    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        text = extract_text_from_pdf(path)
    elif ext == "docx":
        text = extract_text_from_docx(path)
    elif ext == "txt":
        text = extract_text_from_txt(path)
    else:
        return bot.send_message(uid, "❌ صيغة غير مدعومة. أرسل PDF أو DOCX أو TXT.")

    cursor.execute("SELECT major FROM users WHERE user_id = ?", (uid,))
    major = cursor.fetchone()[0] or "عام"

    bot.send_message(uid, "🧠 جاري توليد الاختبارات... الرجاء الانتظار")
    quizzes = generate_quizzes_from_text(text[:3000], major, num_quizzes=3)
    if quizzes:
        send_quizzes_as_polls(uid, quizzes)
        increment_count(uid)
    else:
        bot.send_message(uid, "❌ حدث خطأ أثناء توليد الاختبارات. حاول لاحقًا.")

@bot.message_handler(content_types=['text'])
def handle_text(msg):
    uid = msg.chat.id
    # skip if awaiting major
    if user_states.get(uid) == "awaiting_major":
        return

    if not can_generate(uid):
        return bot.send_message(uid, "⚠️ لقد استنفدت 3 اختبارات مجانية هذا الشهر.")

    text = msg.text.strip()
    cursor.execute("SELECT major FROM users WHERE user_id = ?", (uid,))
    major = cursor.fetchone()[0] or "عام"

    bot.send_message(uid, "🧠 جاري توليد الاختبارات من النص... الرجاء الانتظار")
    quizzes = generate_quizzes_from_text(text[:3000], major, num_quizzes=3)
    if quizzes:
        send_quizzes_as_polls(uid, quizzes)
        increment_count(uid)
    else:
        bot.send_message(uid, "❌ حدث خطأ أثناء توليد الاختبارات. حاول لاحقًا.")

# -------------------------------------------------------------------
#                           Run Bot
# -------------------------------------------------------------------

# واجهة Flask للفحص
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ البوت يعمل الآن"

# بدء البوت

# تشغيل بوت تيليغرام في Thread منفصل
def run_bot():
    print("🤖 Bot polling started...")
    bot.infinity_polling()

threading.Thread(target=run_bot).start()

# تشغيل خادم Flask على المنفذ الذي تحدده Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render يوفر PORT كمتغير بيئة
    app.run(host="0.0.0.0", port=port)
