import os
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
import json
import re

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


# --- إعداد المفاتيح والعمل

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

def generate_gemini_response(prompt: str, chat_id: int) -> str:
    """
    توليد استجابة باستخدام عدة نماذج ذكاء اصطناعي، مع إرسال رسائل توضيحية للمستخدم عبر chat_id.
    """
    timeout_seconds = 45

    # 1️⃣ Google Gemini
    if gemini_model:
        try:
            bot.send_message(chat_id, "🤖 1. محاولة الاتصال بـ Google Gemini...")
            request_options = {"timeout": timeout_seconds}
            response = gemini_model.generate_content(prompt, request_options=request_options)
            if response.text:
                bot.send_message(chat_id, "✅ 1. نجح الاتصال بـ Gemini.")
                return response.text
            else:
                bot.send_message(chat_id, "⚠️ 1. لم يرجع Gemini أي نص. ننتقل إلى الخيار التالي...")
        except Exception as e:
            bot.send_message(chat_id, f"❌ 1. فشل Gemini: {str(e)[:300]}")
            logging.warning(f"❌ Gemini failed: {e}")

    # 2️⃣ Groq (Llama 3)
    if groq_client:
        try:
            bot.send_message(chat_id, "🤖 2. محاولة الاتصال بـ Groq (LLaMA 3)...")
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.7,
                timeout=timeout_seconds
            )
            if chat_completion.choices[0].message.content:
                bot.send_message(chat_id, "✅ 2. نجح الاتصال بـ Groq.")
                return chat_completion.choices[0].message.content
            else:
                bot.send_message(chat_id, "⚠️ 2. لم يرجع Groq أي محتوى. ننتقل إلى الخيار التالي...")
        except Exception as e:
            bot.send_message(chat_id, f"❌ 2. فشل Groq: {str(e)[:300]}")
            logging.warning(f"❌ Groq failed: {e}")

    # 3️⃣ OpenRouter - Mistral
    if OPENROUTER_API_KEY:
        try:
            bot.send_message(chat_id, "🤖 3. محاولة الاتصال بـ OpenRouter (Mistral)...")
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json={
                    "model": "mistralai/mistral-7b-instruct-free",
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=timeout_seconds
            )
            response.raise_for_status()
            result_text = response.json()['choices'][0]['message']['content']
            bot.send_message(chat_id, "✅ 3. نجح الاتصال بـ OpenRouter (Mistral).")
            return result_text
        except Exception as e:
            bot.send_message(chat_id, f"❌ 3. فشل OpenRouter (Mistral): {str(e)[:300]}")
            logging.warning(f"❌ OpenRouter (Mistral) failed: {e}")

    # 4️⃣ OpenRouter - Gemma
    if OPENROUTER_API_KEY:
        try:
            bot.send_message(chat_id, "🤖 4. محاولة الاتصال بـ OpenRouter (Gemma)...")
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json={
                    "model": "google/gemma-7b-it-free",
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=timeout_seconds
            )
            response.raise_for_status()
            result_text = response.json()['choices'][0]['message']['content']
            bot.send_message(chat_id, "✅ 4. نجح الاتصال بـ OpenRouter (Gemma).")
            return result_text
        except Exception as e:
            bot.send_message(chat_id, f"❌ 4. فشل OpenRouter (Gemma): {str(e)[:300]}")
            logging.warning(f"❌ OpenRouter (Gemma) failed: {e}")

    # 5️⃣ Cohere
    if cohere_client:
        try:
            bot.send_message(chat_id, "🤖 5. محاولة الاتصال بـ Cohere...")
            response = cohere_client.chat(model='command-r', message=prompt)
            bot.send_message(chat_id, "✅ 5. نجح الاتصال بـ Cohere.")
            return response.text
        except Exception as e:
            bot.send_message(chat_id, f"❌ 5. فشل Cohere: {str(e)[:300]}")
            logging.warning(f"❌ Cohere failed: {e}")

    # 🚫 كل النماذج فشلت
    logging.error("❌ All models failed.")
    bot.send_message(chat_id, "⚠️ فشلت جميع المحاولات للاتصال بالنماذج. الرجاء المحاولة لاحقًا أو التواصل مع المطور.")
    return ""

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

def extract_json_from_string(text: str) -> str:
    """
    Extracts a JSON string from a text that might contain markdown code blocks or other text.
    """
    # البحث عن بلوك JSON داخل ```json ... ```
    match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if match:
        return match.group(1).strip()

    # إذا لم يجد بلوك، ابحث عن أول '{' أو '[' وآخر '}' أو ']'
    start = -1
    end = -1
    
    # البحث عن بداية القائمة أو الكائن
    first_brace = text.find('{')
    first_bracket = text.find('[')
    
    if first_brace == -1:
        start = first_bracket
    elif first_bracket == -1:
        start = first_brace
    else:
        start = min(first_brace, first_bracket)

    # إذا لم يتم العثور على بداية، أرجع النص الأصلي
    if start == -1:
        return text

    # البحث عن نهاية القائمة أو الكائن
    last_brace = text.rfind('}')
    last_bracket = text.rfind(']')
    end = max(last_brace, last_bracket)

    # إذا تم العثور على بداية ونهاية، أرجع ما بينهما
    if end > start:
        return text[start:end+1].strip()
        
    # كخيار أخير، أرجع النص كما هو
    return text
    
def generate_quizzes_from_text(text: str, major: str, chat_id: int, num_quizzes: int = 10):
    prompt = (
        f"You are an AI quiz generator. Generate a JSON array of {num_quizzes} quiz questions "
        f"suitable for students majoring in {major}, based on the following content.\n\n"
        "Each question must be an object with:\n"
        "- 'question': the question string\n"
        "- 'options': a list of exactly 4 answer options\n"
        "- 'correct_index': the index (0-3) of the correct answer in the options list\n\n"
        "⚠️ Important Instructions:\n"
        "- ONLY return a raw JSON array. No markdown, no explanation, no formatting.\n"
        "- Do not include any introductory or closing text.\n"
        "- Ensure the JSON is valid and parsable.\n\n"
        f"Content:\n{text}"
    )

    raw_response = generate_gemini_response(prompt, chat_id)
    
    # --- التعديل يبدأ هنا ---
    # 1. تنظيف الاستجابة لاستخراج الـ JSON
    clean_json_str = extract_json_from_string(raw_response)
    
    # 2. التحقق مما إذا كانت الاستجابة فارغة بعد التنظيف
    if not clean_json_str:
        logging.error(f"❌ JSON extraction failed. Raw output was:\n{raw_response}")
        return [] # أرجع قائمة فارغة بدلاً من رسالة خطأ

    try:
        # 3. محاولة تحليل السلسلة النظيفة
        quizzes_json = json.loads(clean_json_str)
        quizzes = []

        for item in quizzes_json:
            q = item.get("question", "").strip()
            opts = item.get("options", [])
            corr = item.get("correct_index", -1)

            if isinstance(q, str) and q and isinstance(opts, list) and len(opts) == 4 and isinstance(corr, int) and 0 <= corr < 4:
                quizzes.append((q, [str(opt).strip() for opt in opts], corr))
            else:
                logging.warning(f"❌ Skipping invalid question structure: {item}")

        return quizzes

    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON parsing failed: {e}\nCleaned string was:\n{clean_json_str}\nRaw output was:\n{raw_response}")
        return [] # أرجع قائمة فارغة عند الفشل
    # --- التعديل ينتهي هنا ---

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
    uid = msg.from_user.id
    if not can_generate(uid):
        return bot.send_message(uid, "⚠️ لقد استنفدت 3 اختبارات مجانية هذا الشهر.")

    file_info = bot.get_file(msg.document.file_id)
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

    if file_info.file_size > MAX_FILE_SIZE:
        return bot.send_message(uid, "❌ الملف كبير جدًا. الحد الأقصى هو 5 ميجابايت.")
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
    quizzes = generate_quizzes_from_text(text[:3000], major, num_quizzes=10)
    if quizzes and len(quizzes) > 0:
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
    quizzes = generate_quizzes_from_text(text[:3000], major, num_quizzes=10)
    if quizzes and len(quizzes) > 0:
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
