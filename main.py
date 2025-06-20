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
import requests
import cohere
from groq import Groq

# متغيرات البيئة
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROUP_ID = int(os.getenv("GROUP_ID"))
ADMIN_ID = int(os.getenv("ADMIN_ID"))
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

bot = telebot.TeleBot(BOT_TOKEN)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# --- إعداد المفاتيح والعملاء ---

# استدعاء المفاتيح من متغيرات البيئة
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 1. إعداد Google Gemini
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        logging.info("✅ 1. Gemini configured successfully")
    except Exception as e:
        logging.warning(f"⚠️ Could not configure Gemini: {e}")

# 2. إعداد Groq
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logging.info("✅ 2. Groq configured successfully")
    except Exception as e:
        logging.warning(f"⚠️ Could not configure Groq: {e}")

# 3. إعداد OpenRouter (سيتم استخدامه لنموذجين مختلفين)
if OPENROUTER_API_KEY:
    logging.info("✅ 3. OpenRouter is ready")

# 4. إعداد Cohere
cohere_client = None
if COHERE_API_KEY:
    try:
        cohere_client = cohere.Client(COHERE_API_KEY)
        logging.info("✅ 4. Cohere configured successfully")
    except Exception as e:
        logging.warning(f"⚠️ Could not configure Cohere: {e}")


# --- الدالة الموحدة لتوليد الردود ---

def generate_gemini_response(prompt: str) -> str:
    """
    Tries to generate a response by attempting a chain of services for maximum reliability:
    1. Google Gemini (Primary)
    2. Groq (Fastest Fallback)
    3. OpenRouter w/ Mistral (Diverse Fallback)
    4. OpenRouter w/ Gemma (Second Diverse Fallback)
    5. Cohere (Final Fallback)
    """

    # 1️⃣ المحاولة الأولى: Google Gemini (الأساسي والمتوازن)
    if gemini_model:
        try:
            logging.info("Attempting request with: 1. Google Gemini...")
            response = gemini_model.generate_content(prompt)
            if response.text:
                logging.info("✅ Success with Gemini.")
                return response.text
            else:
                logging.warning("❌ Gemini returned no text. Trying fallback...")
        except Exception as e:
            logging.warning(f"❌ Gemini failed: {e}. Trying fallback...")

    # 2️⃣ المحاولة الثانية: Groq (للأداء فائق السرعة)
    if groq_client:
        try:
            logging.info("Attempting request with: 2. Groq (Llama 3)...")
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192", # نموذج قوي وسريع جداً
                temperature=0.7
            )
            if chat_completion.choices[0].message.content:
                logging.info("✅ Success with Groq.")
                return chat_completion.choices[0].message.content
            else:
                logging.warning("❌ Groq returned no text. Trying fallback...")
        except Exception as e:
            logging.warning(f"❌ Groq failed: {e}. Trying fallback...")

    # 3️⃣ المحاولة الثالثة: OpenRouter (مع نموذج MistralAI)
    if OPENROUTER_API_KEY:
        try:
            logging.info("Attempting request with: 3. OpenRouter (Mistral)...")
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json={
                    "model": "mistralai/mistral-7b-instruct-free", # نموذج Mistral مجاني وفعال
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            response.raise_for_status()
            result_text = response.json()['choices'][0]['message']['content']
            logging.info("✅ Success with OpenRouter (Mistral).")
            return result_text
        except Exception as e:
            logging.warning(f"❌ OpenRouter (Mistral) failed: {e}. Trying fallback...")

    # 4️⃣ المحاولة الرابعة: OpenRouter (مع نموذج Google Gemma)
    if OPENROUTER_API_KEY:
        try:
            logging.info("Attempting request with: 4. OpenRouter (Gemma)...")
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json={
                    "model": "google/gemma-7b-it-free", # خيار احتياطي مجاني آخر
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            response.raise_for_status()
            result_text = response.json()['choices'][0]['message']['content']
            logging.info("✅ Success with OpenRouter (Gemma).")
            return result_text
        except Exception as e:
            logging.warning(f"❌ OpenRouter (Gemma) failed: {e}. Trying fallback...")

    # 5️⃣ المحاولة الخامسة: Cohere (الخيار الاحتياطي الأخير)
    if cohere_client:
        try:
            logging.info("Attempting request with: 5. Cohere...")
            response = cohere_client.chat(model='command-r', message=prompt)
            logging.info("✅ Success with Cohere.")
            return response.text
        except Exception as e:
            logging.error(f"❌ Cohere failed: {e}. All fallbacks exhausted.")

    # في حال فشل جميع المحاولات
    logging.error("❌ All API providers failed.")
    return "⚠️ للأسف، فشل الاتصال بجميع نماذج الذكاء الاصطناعي المتاحة. يرجى المحاولة مرة أخرى لاحقًا."


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
# أضف هذه الدالة في قسم Text Extraction & OCR
def extract_text_from_docx(path: str) -> str:
    try:
        doc = docx.Document(path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        logging.error(f"Error extracting DOCX text: {e}")
        return ""

# ويجب أيضاً تعريف دالة لملفات txt
def extract_text_from_txt(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error extracting TXT text: {e}")
        return ""

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
