import os
import sqlite3
import time # <--- أضف هذا السطر
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

session = {}  # <--- أضف هذا السطر

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
    Tries to generate a response by attempting a chain of services silently.
    It logs errors for the developer but does not send progress messages to the user.
    """
    timeout_seconds = 45

    # 1️⃣ OpenRouter - Nous Hermes 2 (أفضل دعم للعربية)
    if OPENROUTER_API_KEY:
        try:
            logging.info("Attempting request with: 1. OpenRouter (Nous Hermes 2)...")
            headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://t.me/Oiuhelper_bot",  # ← غيّر هذا إلى رابط البوت
            "X-Title": "AI Quiz Bot"
            }
            model_identifier = "nousresearch/nous-hermes-2-mistral:free"
            response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
                json={
                "model": model_identifier,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=timeout_seconds
            )
            response.raise_for_status()
            result_text = response.json()['choices'][0]['message']['content']
            logging.info("✅ Success with OpenRouter (Nous Hermes 2).")
            return result_text
        except Exception as e:
            logging.warning(f"❌ OpenRouter (Nous Hermes 2) failed: {e}")

    # 2️⃣ Groq (LLaMA 3)
    if groq_client:
        try:
            logging.info("Attempting request with: 2. Groq (LLaMA 3)...")
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.7,
                timeout=timeout_seconds
            )
            if chat_completion.choices[0].message.content:
                logging.info("✅ Success with Groq.")
                return chat_completion.choices[0].message.content
            else:
                logging.warning("❌ Groq returned no text. Trying fallback...")
        except Exception as e:
            logging.warning(f"❌ Groq failed: {e}")

    # 3️⃣ OpenRouter - Gemma
    if OPENROUTER_API_KEY:
        try:
            logging.info("Attempting request with: 3. OpenRouter (Gemma)...")
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://t.me/Oiuhelper_bot",  # Replace with your bot's link
                "X-Title": "AI Quiz Bot"
            }
            model_identifier = "google/gemma-7b-it:free"
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={"model": model_identifier, "messages": [{"role": "user", "content": prompt}]},
                timeout=timeout_seconds
            )
            response.raise_for_status()
            result_text = response.json()['choices'][0]['message']['content']
            logging.info("✅ Success with OpenRouter (Gemma).")
            return result_text
        except Exception as e:
            logging.warning(f"❌ OpenRouter (Gemma) failed: {e}")

    # 4️⃣ Google Gemini
    if gemini_model:
        try:
            logging.info("Attempting request with: 4. Google Gemini...")
            request_options = {"timeout": timeout_seconds}
            response = gemini_model.generate_content(prompt, request_options=request_options)
            if response.text:
                logging.info("✅ Success with Gemini.")
                return response.text
            else:
                logging.warning("❌ Gemini returned no text. Trying fallback...")
        except Exception as e:
            logging.warning(f"❌ Gemini failed: {e}")

    # 5️⃣ Cohere
    if cohere_client:
        try:
            logging.info("Attempting request with: 5. Cohere...")
            response = cohere_client.chat(model='command-r', message=prompt)
            logging.info("✅ Success with Cohere.")
            return response.text
        except Exception as e:
            logging.warning(f"❌ Cohere failed: {e}")

    # 🚫 All models failed
    logging.error("❌ All API providers failed. Returning empty string.")
    return ""


def generate_smart_response(prompt: str) -> str:
    """
    Tries to generate a response by attempting a chain of services silently.
    It logs errors for the developer but does not send progress messages to the user.
    """
    timeout_seconds = 45


    #  1️⃣ Cohere
    if cohere_client:
        try:
            logging.info("Attempting request with: 5. Cohere...")
            response = cohere_client.chat(model='command-r', message=prompt, temperature=0.8)
            logging.info("✅ Success with Cohere.")
            return response.text
        except Exception as e:
            logging.warning(f"❌ Cohere failed: {e}")



    # 2️⃣ Google Gemini
    if gemini_model:
        try:
            logging.info("Attempting request with: 4. Google Gemini...")
            request_options = {"timeout": timeout_seconds}
            response = gemini_model.generate_content(prompt, request_options=request_options, temperature=0.8)
            if response.text:
                logging.info("✅ Success with Gemini.")
                return response.text
            else:
                logging.warning("❌ Gemini returned no text. Trying fallback...")
        except Exception as e:
            logging.warning(f"❌ Gemini failed: {e}")


    #  3️⃣  Groq (LLaMA 3)
    if groq_client:
        try:
            logging.info("Attempting request with: 2. Groq (LLaMA 3)...")
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.8,
                timeout=timeout_seconds
            )
            if chat_completion.choices[0].message.content:
                logging.info("✅ Success with Groq.")
                return chat_completion.choices[0].message.content
            else:
                logging.warning("❌ Groq returned no text. Trying fallback...")
        except Exception as e:
            logging.warning(f"❌ Groq failed: {e}")

    # 4️⃣# 5️⃣ OpenRouter - Gemma
    if OPENROUTER_API_KEY:
        try:
            logging.info("Attempting request with: 3. OpenRouter (Gemma)...")
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://t.me/Oiuhelper_bot",  # Replace with your bot's link
                "X-Title": "AI Quiz Bot"
            }
            model_identifier = "google/gemma-7b-it:free"
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={"model": model_identifier, "messages": [{"role": "user", "content": prompt}]},
                timeout=timeout_seconds
            )
            response.raise_for_status()
            result_text = response.json()['choices'][0]['message']['content']
            logging.info("✅ Success with OpenRouter (Gemma).")
            return result_text
        except Exception as e:
            logging.warning(f"❌ OpenRouter (Gemma) failed: {e}")

    # 🚫 All models failed
    logging.error("❌ All API providers failed. Returning empty string.")
    return ""


def translate_text(text, source='en', target='ar'):
    url = 'https://libretranslate.de/translate'
    payload = {
        'q': text,
        'source': source,
        'target': target,
        'format': 'text'
    }
    try:
        response = requests.post(url, data=payload)
        return response.json()['translatedText']
    except Exception as e:
        print("ترجمة فشلت:", e)
        return text  # fallback


from flask import Flask, render_template, session, request, redirect, url_for



# -------------------------------------------------------------------
#                  Logging & Database Setup
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

conn   = sqlite3.connect("quiz_users.db", check_same_thread=False)
cursor = conn.cursor()

# جدول المستخدمين
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id     INTEGER PRIMARY KEY,
    major       TEXT,
    native_lang TEXT DEFAULT 'ar',
    quiz_count  INTEGER DEFAULT 0,
    last_reset  TEXT
)
""")

# جدول الأسئلة المقترحة من المستخدمين للعبة الاستنتاج
cursor.execute("""
CREATE TABLE IF NOT EXISTS inference_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    options TEXT NOT NULL,         -- سيتم تخزينها كسلسلة JSON
    correct_index INTEGER NOT NULL,
    submitted_by INTEGER,
    approved INTEGER DEFAULT 0
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS game_attempts (
    user_id INTEGER,
    game_type TEXT,
    date TEXT,
    PRIMARY KEY (user_id, game_type)
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS recent_questions (
    user_id INTEGER,
    game_type TEXT,
    question TEXT,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        


def parse_ai_json(raw_text: str) -> dict | None:
    """
    1) يحوّل هاربات Unicode إلى نص عربي.
    2) يقطّع أول كتلة JSON موجودة داخل النص إن لم يكن الرد نقيًّا.
    3) يحاول json.loads عدة مرات.
    4) يتحقق من بنية الناتج قبل الإرجاع.
    """
    # 1. فكُّ هاربات Unicode (\u0627 → ا)
    def _unescape(match):
        code = match.group(1)
        return chr(int(code, 16))
    text = re.sub(r'\\u([0-9A-Fa-f]{4})', _unescape, raw_text)

    # 2. اجتزء أول كتلة JSON (من { إلى })
    m = re.search(r'\{[\s\S]*\}', text)
    json_text = m.group(0) if m else text

    # 3. حاول التحميل
    for attempt in (json_text, text):
        try:
            data = json.loads(attempt)
            break
        except json.JSONDecodeError:
            data = None
    if not data:
        return None

    # 4. التحقق من بنية الـ dict
    if not all(k in data for k in ("question", "options", "correct_index")):
        return None

    # 5. التأكد من أن options قائمة وصالحة
    if not isinstance(data["options"], list) or len(data["options"]) < 2:
        return None

    # 6. التأكد من correct_index
    ci = data["correct_index"]
    if not isinstance(ci, int) or ci < 0 or ci >= len(data["options"]):
        return None

    return data

def generate_game(prompt, user_id=0, translate_all=False, translate_question=False):
    if user_id == ADMIN_ID or can_generate(user_id):  # <-- التحقق هنا
        raw_response = generate_smart_response(prompt)
    else:
        raw_response = generate_gemini_response(prompt)
        
    game_data = parse_ai_json(raw_response)

    if not game_data:
        raise ValueError("فشل استخراج بيانات اللعبة")

    if translate_all:
        # ترجمة السؤال
        if 'question' in game_data:
            game_data['question'] = translate_text(game_data['question'], source='en', target='ar')

        # ترجمة كل الخيارات
        if 'options' in game_data and isinstance(game_data['options'], list):
            game_data['options'] = [
                translate_text(option, source='en', target='ar') for option in game_data['options']
            ]

    elif translate_question:
        # ترجمة السؤال فقط
        if 'question' in game_data:
            game_data['question'] = translate_text(game_data['question'], source='en', target='ar')

    return game_data

import genanki
import time
import uuid

def save_cards_to_apkg(cards: list, filename='anki_flashcards.apkg', deck_name="My Flashcards"):
    model = genanki.Model(
        1607392319,
        'Simple Model',
        fields=[
            {'name': 'Front'},
            {'name': 'Back'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '{{Front}}',
                'afmt': '{{FrontSide}}<hr id="answer">{{Back}}',
            },
        ]
    )

    deck = genanki.Deck(
        deck_id=int(str(uuid.uuid4().int)[:9]),  # لتوليد رقم deck عشوائي وفريد
        name=deck_name
    )

    seen = set()
    for card in cards:
        front = card.get('front', '').strip()
        back = card.get('back', '').strip()
        if front and back and front not in seen:
            note = genanki.Note(model=model, fields=[front, back])
            deck.add_note(note)
            seen.add(front)

    genanki.Package(deck).write_to_file(filename)
    return filename

    
# -------------------------------------------------------------------
#                     Quota Management
# -------------------------------------------------------------------
def add_recent_question(user_id, game_type, question):
    with sqlite3.connect("quiz_users.db") as conn:
        cursor = conn.cursor()
        
        # إدخال السؤال الجديد
        cursor.execute("""
        INSERT INTO recent_questions (user_id, game_type, question) 
        VALUES (?, ?, ?)
        """, (user_id, game_type, question))
        
        # حذف الأقدم إذا تجاوز 10 أسئلة
        cursor.execute("""
        DELETE FROM recent_questions
        WHERE user_id = ? AND game_type = ?
        AND question NOT IN (
            SELECT question FROM recent_questions
            WHERE user_id = ? AND game_type = ?
            ORDER BY added_at DESC
            LIMIT 10
        )
        """, (user_id, game_type, user_id, game_type))

        conn.commit()

def get_recent_questions(user_id, game_type):
    with sqlite3.connect("quiz_users.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT question FROM recent_questions
        WHERE user_id = ? AND game_type = ?
        ORDER BY added_at DESC
        LIMIT 10
        """, (user_id, game_type))
        rows = cursor.fetchall()
        return [row[0] for row in rows]


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
    # --- بداية التعديل ---
    # إذا كان المستخدم هو الأدمن، اسمح له دائمًا بالتوليد
    if user_id == ADMIN_ID:
        return True
    # --- نهاية التعديل ---

    reset_if_needed(user_id)
    cursor.execute("SELECT quiz_count FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    # تأكد من أن المستخدم موجود قبل محاولة الوصول إلى العد
    if not row:
        return True # مستخدم جديد، يمكنه التوليد
    count = row[0]
    return count < 6

def increment_count(user_id: int):
    # --- بداية التعديل ---
    # لا تقم بزيادة العداد إذا كان المستخدم هو الأدمن
    if user_id == ADMIN_ID:
        bot.send_message(ADMIN_ID, "✨ (وضع الأدمن: لم يتم احتساب هذه المحاولة)")
        return
    # --- نهاية التعديل ---
    
    cursor.execute("UPDATE users SET quiz_count = quiz_count + 1 WHERE user_id = ?", (user_id,))
    conn.commit()

from datetime import date

def can_play_game_today(user_id: int, game_type: str) -> bool:
    if str(user_id) == str(ADMIN_ID):  # مقارنة آمنة لأن ADMIN_ID أحيانًا يكون str
        return True

    today = str(date.today())
    cursor.execute(
        "SELECT 1 FROM game_attempts WHERE user_id = ? AND game_type = ? AND date = ?",
        (user_id, game_type, today)
    )
    return cursor.fetchone() is None

def record_game_attempt(user_id: int, game_type: str):
    if str(user_id) == str(ADMIN_ID):
        return  # لا تسجل للأدمن

    today = str(date.today())
    cursor.execute(
        "INSERT OR REPLACE INTO game_attempts(user_id, game_type, date) VALUES (?, ?, ?)",
        (user_id, game_type, today)
    )
    conn.commit()
from collections import defaultdict

# تخزين مؤقت في الذاكرة
game_states = defaultdict(dict)  # {user_id: {game_type: count}}

def get_question_count(user_id, game_type):
    return game_states.get(user_id, {}).get(game_type, 0)

def increment_question_count(user_id, game_type):
    game_states[user_id][game_type] = game_states.get(user_id, {}).get(game_type, 0) + 1
    
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
    
def generate_quizzes_from_text(content: str, major: str, user_id: int, num_quizzes: int = 10):  # <-- أضف user_id
    prompt = (
    f"You are a strict AI quiz generator. Your only task is to generate a JSON array of {num_quizzes} quiz questions "
    f"that are based **strictly and only** on the information explicitly stated in the following content.\n\n"
    "❗️Important Rules:\n"
    "- DO NOT invent, infer, or assume any information not clearly mentioned in the text.\n"
    "- If a concept is not explained or mentioned clearly in the content, DO NOT create a question about it.\n"
    "- Stay fully inside the boundaries of the content.\n"
    "- Every question must test **recall** or **recognition** from the provided text only, not general knowledge.\n\n"
    "Each question must be an object with:\n"
    "- 'question': the question string\n"
    "- 'options': a list of exactly 4 answer options\n"
    "- 'correct_index': the index (0-3) of the correct answer in the options list\n"
    "- 'explanation': short sentence to explain **why this is the correct answer**, max 2 lines\n\n"
    "⚠️ Format Instructions:\n"
    "- ONLY return a raw JSON array. No markdown, no explanation, no formatting.\n"
    "- Do not include any introductory or closing text.\n"
    "- Ensure the JSON is valid and parsable.\n\n"
    f"Content:\n{content}"
    )

    # تحديد الدالة بناءً على صلاحية المستخدم
    if user_id == ADMIN_ID or can_generate(user_id):  # <-- التحقق هنا
        raw_response = generate_smart_response(prompt)
    else:
        raw_response = generate_gemini_response(prompt)
    
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
            expl = item.get("explanation", "").strip()

            if isinstance(q, str) and q and isinstance(opts, list) and len(opts) == 4 and isinstance(corr, int) and 0 <= corr < 4:
                quizzes.append((q, [str(opt).strip() for opt in opts], corr, expl))
            else:
                logging.warning(f"❌ Skipping invalid question structure: {item}")

        return quizzes

    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON parsing failed: {e}\nCleaned string was:\n{clean_json_str}\nRaw output was:\n{raw_response}")
        return [] # أرجع قائمة فارغة عند الفشل
    # --- التعديل ينتهي هنا ---


def generate_anki_cards_from_text(content: str, major: str = "General", user_id: int = 0, num_cards: int = 15) -> tuple:
    for attempt in range(3):  # تجربة حتى 3 مرات
        prompt = f"""
You are an AI assistant specialized in creating study flashcards.

🎯 Task:
Extract the most important {num_cards} points from the following content, and convert each into an **Anki-style flashcard**.

🔹 Rules:
- Each flashcard must include:
  - "front": a short question or hint.
  - "back": the detailed answer or explanation.
  - "tag": (optional) topic label like Grammar, Biology, Logic, etc.
- The front must be phrased to encourage recall (e.g. "What is...", "Define...", "How does...").
- Don't use Markdown, just clean plain text.
- Keep the cards diverse and helpful.
- Output must be a valid JSON **object** with two keys: "title" and "cards".

🚫 Important:
- Do NOT generate multiple choice or true/false questions.
- Only generate flashcards suitable for Anki with a front and a back.
- The flashcards must be written in the same language as the input content. If the content is in Arabic, answer in Arabic. If English, answer in English.

📘 Content to process (field: {major}):
{content}

✅ Example output format:
{{
  "title": "Basics of Organic Chemistry",
  "cards": [
    {{
      "front": "What is the function of mitochondria?",
      "back": "It is the powerhouse of the cell.",
      "tag": "Biology"
    }},
    {{
      "front": "ما هي الاستعارة؟",
      "back": "الاستعارة هي استخدام الكلمة في غير معناها الحقيقي لعلاقة مع قرينة مانعة.",
      "tag": "Literature"
    }}
  ]
}}
"""
        if user_id == ADMIN_ID or can_generate(user_id):  # <-- التحقق هنا
            raw_output = generate_smart_response(prompt)
        else:
            raw_output = generate_gemini_response(prompt)
            
        clean_json = extract_json_from_string(raw_output)

        try:
            data = json.loads(clean_json)
            title = data.get("title", "بطاقات تعليمية")
            card_list = data.get("cards", [])

            cards = []
            for item in card_list:
                front = item.get("front") or item.get("question")
                back = item.get("back") or item.get("answer")

                if isinstance(front, str) and isinstance(back, str) and front.strip() and back.strip():
                    cards.append({"front": front.strip(), "back": back.strip()})
                else:
                    logging.warning(f"❌ Skipping invalid card: {item}")

            if len(cards) >= 5:
                return cards, title

        except json.JSONDecodeError as e:
            logging.error(f"❌ Failed to parse Anki cards: {e}\nClean JSON:\n{clean_json}\nRaw:\n{raw_output}")

    return [], "بطاقات تعليمية"   

    


# -------------------------------------------------------------------
#                 games
# -------------------------------------------------------------------
import random

topics = [
    "حياة الطالب", "تخطيط السفر", "مشاريع جماعية", "مقابلات العمل",
    "الضغط الزمني", "مواقف عاطفية", "استخدام التكنولوجيا", "قرارات مالية",
    "صراعات الفريق", "تحديد الأهداف"
]

def generate_vocabulary_game(user_id, major, native_lang="Arabic"):
    rand = random.randint(1000, 9999)
    recent = get_recent_questions(user_id, "vocab")
    recent_prompt = "\n".join(f"- {q}" for q in recent)
    
    prompt = f"""  
You are an AI vocabulary quiz creator.  
Generate one vocabulary question for a student majoring in {major}.
- Vocabulary should be relevant to real life or academic use and not an uncommon Vocabulary.
- Show the meaning of an English word in English 
- Provide 4 English words as options  
- Only ONE option should be correct.  
- Don't explain anything. Just give raw JSON.

Example:
{{
  "question": "Question",
  "options": ["Option", "Option", "Option", "Option"],
  "correct_index": 0
}}

Use this seed to diversify the question: {rand}
❌ Avoid repeating or paraphrasing these questions:
{recent_prompt}
"""
    q = generate_game(prompt)

    # حفظ السؤال الجديد
    add_recent_question(user_id, "speed", q["question"])
    return q

def generate_speed_challenge(user_id, major, native_lang="Arabic"):
    rand = random.randint(1000, 9999)
    recent = get_recent_questions(user_id, "speed")
    recent_prompt = "\n".join(f"- {q}" for q in recent)
    
    prompt = f"""
You are a quiz bot.

Generate a **fun, fast-answer quiz** for a student in {major}.

Requirements:
- The question must be in English.
- The 4 options must be in English.
- Use fun and fast general knowledge topics (e.g. logic, daily life trivia, or language puzzles). Avoid repeating the same categories.
- Keep it simple and not too academic.
- Return raw JSON only.
- No explanation.
- Use this seed to increase randomness: {rand}
❌ Avoid repeating or paraphrasing these questions:
{recent_prompt}

Example output:
{{
  "question": "Question?",
  "options": ["Option", "Option", "Option", "Option"],
  "correct_index": 0
}}
"""
    q = generate_game(prompt, translate_question=True)

    # حفظ السؤال الجديد
    add_recent_question(user_id, "speed", q["question"])
    return q
    

# ★ لعبة الاخطاء الشائعة
def generate_common_mistakes_game(user_id, major, native_lang="Arabic"):
    rand = random.randint(1000, 9999)
    recent = get_recent_questions(user_id, "mistakes")
    recent_prompt = "\n".join(f"- {q}" for q in recent)
    
    prompt = f"""
You are an educational game generator.

Your task:
- Generate a multiple-choice question highlighting a **common mistake** in the field of {major}.
- The question must be in English.
- The **options must be in English**.
- Provide **4 options** only, with one correct.
- Don't explain.
- Return only raw JSON.

❌ Avoid repeating or paraphrasing these questions:
{recent_prompt}
Use this random seed to diversify the question: {rand}

Example output:
{{
  "question": "Which sentence is grammatically incorrect?",
  "options": ["He go to school every day.", "She plays the piano.", "They are studying now.", "I have finished my homework."],
  "correct_index": 0
}}
"""
    q = generate_game(prompt, translate_question=True)

    # حفظ السؤال الجديد
    add_recent_question(user_id, "speed", q["question"])
    return q


def generate_inference_game(user_id, major, native_lang="Arabic"):
    rand = random.randint(1000, 9999)
    recent = get_recent_questions(user_id, "inference")
    recent_prompt = "\n".join(f"- {q}" for q in recent)
    
    random_topic = random.choice(topics)
    prompt = f"""
You are an AI-powered life skills test creator.

Generate a **new and unique** question that develops one of the following skills:  
- Critical thinking  
- Emotional intelligence  
- Time management  
- Self-awareness  
- Decision making  
- Problem solving  
- Logic  
- Pattern recognition  
- Mental map understanding  

🔹 **Requirements**:  
- Write the **question in Arabic**  
- Write **all options in Arabic**  
- Use a realistic scenario or student-life context related to: **{random_topic}**  
- Provide **exactly 4 options**, with **one correct answer**  
- **Never repeat** past examples or add explanations  
- Make the question **engaging and clever**  
- Incorporate variability using this random number: **{rand}**  
- the options should be as short as possible but understandable
❌ Avoid repeating or paraphrasing these questions:
{recent_prompt}
🔸 Return **JSON-only output** (no additional text).  

Example (Johnson’s format):  
{{
  "question": "Question",  
  "options": ["Options", "Option", "Option", "Option"],  
  "correct_index": 2  
}}  
"""
    q = generate_game(prompt, translate_question=True)

    # حفظ السؤال الجديد
    add_recent_question(user_id, "speed", q["question"])
    return q
    
# ----------------------------------
# ------------- inference review -------------------------------------------------------------------


def review_inference_question_with_ai(question_text: str, options: list[str], correct_index: int) -> bool:
    prompt = f"""
You are an AI educational assistant.

A student submitted the following inference question. Review it and decide if it's valid:
- Is the question clear and meaningful?
- Are the 4 options distinct and related to the question?
- Is there **one and only one correct answer**?

Respond only with YES or NO.

Question: {question_text}
Options: {options}
Correct index: {correct_index}
"""
    response = generate_smart_response(prompt).strip().lower()
    return "yes" in response


def process_pending_inference_questions():
    cursor.execute("SELECT id, question, options, correct_index FROM inference_questions WHERE approved = 0")
    pending = cursor.fetchall()

    for row in pending:
        qid, qtext, options_json, correct_index = row
        try:
            options = json.loads(options_json)
        except:
            continue  # تجاهل الأسئلة ذات التنسيق الخاطئ

        if review_inference_question_with_ai(qtext, options, correct_index):
            cursor.execute("UPDATE inference_questions SET approved = 1 WHERE id = ?", (qid,))
        else:
            cursor.execute("DELETE FROM inference_questions WHERE id = ?", (qid,))

    conn.commit()

def send_quizzes_as_polls(chat_id: int, quizzes: list):
    """
    Sends a list of quizzes to a user as separate Telegram polls.
    :param quizzes: A list of quiz tuples: (question, options, correct_index, explanation)
    """
    bot.send_message(chat_id, f"تم تجهيز {len(quizzes)} سؤالًا. استعد للاختبار!")
    time.sleep(2)

    for i, quiz_data in enumerate(quizzes):
        try:
            question, options, correct_index, explanation = quiz_data

            question_text = f"❓ السؤال {i+1}:\n\n{question}"
            if len(question_text) > 300:
                question_text = question_text[:297] + "..."

            if not explanation:
                explanation = f"الإجابة الصحيحة: {options[correct_index]}"

            bot.send_poll(
                chat_id=chat_id,
                question=question_text,
                options=options,
                type='quiz',
                correct_option_id=correct_index,
                is_anonymous=False,
                explanation=explanation[:200]  # ⚠️ حد تيليجرام
            )

            time.sleep(1)

        except Exception as e:
            logging.error(f"Error sending poll: {e}")
            bot.send_message(chat_id, f"⚠️ خطأ في إرسال السؤال رقم {i+1}.")
            continue

    bot.send_message(chat_id, "🎉 انتهى الاختبار! بالتوفيق.")


# -------------------------------------------------------------------
#                  Telegram Bot Handlers
# -------------------------------------------------------------------

@bot.message_handler(commands=['start'])
def cmd_start(msg):
    if msg.chat.type != "private":
        return  # تجاهل الرسائل في المجموعات
    keyboard = InlineKeyboardMarkup(row_width=2)
    buttons = [
        InlineKeyboardButton("📝 توليد اختبار", callback_data="go_generate"),
        InlineKeyboardButton("📚 مراجعة سريعة", callback_data="soon_review"),
        InlineKeyboardButton("📄 ملخص PDF", callback_data="soon_summary"),
        InlineKeyboardButton("🧠 بطاقات Anki", callback_data="anki"),
        InlineKeyboardButton("🎮 ألعاب تعليمية", callback_data="go_games"),
        InlineKeyboardButton("⚙️ حسابي", callback_data="soon_account"),
    ]
    keyboard.add(*buttons)

    bot.send_message(
        msg.chat.id,
        "👋 أهلا بك في *TestGenie* ✨\n\n"
        "🎯  أدوات تعليمية ذكية بين يديك:\n"
        "- اختبارات من ملفاتك\n"
        "- بطاقات مراجعة (Anki)\n"
        "- ملخصات PDF/Word _(قريباً)_\n"
        "- ألعاب تعليمية \n\n"
        " 📌 كل ما تحتاجه لتتعلّم بذكاء... بين يديك الآن..\n\n"
        "اختر ما يناسبك و إبدأ الآن! 👇",
        reply_markup=keyboard
        parse_mode="Markdown"
    )

@bot.callback_query_handler(func=lambda c: True)
def handle_main_menu(c):
    if c.message.chat.type != "private":
        return
    # ردود خاطئة عشوائية تظهر للمستخدم
    wrong_responses = [
        "❌ خطأ! جرب مجددًا 😉\n✅ الصحيح: {correct}",
        "🚫 للأسف، ليست الصحيحة!\n✅ الجواب: {correct}",
        "😅 ليست الإجابة الصحيحة، الجواب هو: {correct}",
        "❌ لا، حاول مرة أخرى!\n✔️ الصحيح هو: {correct}"
    ]
    uid = c.from_user.id
    data = c.data
    chat_id = c.message.chat.id
    message_id = c.message.message_id

    # معالجة القائمة الرئيسية
    if data == "go_generate":
        keyboard = InlineKeyboardMarkup()
        buttons = [
            ("🩺 الطب", "major_الطب"),
            ("🛠️ الهندسة", "major_الهندسة"),
            ("💊 الصيدلة", "major_الصيدلة"),
            ("🗣️ اللغات", "major_اللغات"),
            ("❓ غير ذلك...", "major_custom"),
        ]
        for text, data_btn in buttons:
            keyboard.add(InlineKeyboardButton(text, callback_data=data_btn))
        keyboard.add(InlineKeyboardButton("⬅️ رجوع", callback_data="go_back_home"))

        bot.edit_message_text(
            "🎯 هذا البوت يساعدك على توليد اختبارات ذكية من ملفاتك الدراسية أو النصوص.\n"
            "📌 متاح لك 3 اختبارات مجانية شهريًا.\n\n"
            "اختر تخصصك للبدء 👇", 
            chat_id=chat_id,
            message_id=message_id,
            reply_markup=keyboard
        )
    elif data == "anki":
        bot.answer_callback_query(c.id)
        bot.send_message(uid, "📄 أرسل الآن ملف PDF أو Word أو نصًا عاديًا لتوليد بطاقات المراجعة (Anki).")
        user_states[uid] = "awaiting_anki_file"  # ← تحديد حالة المستخدم
        
    elif data == "go_games":
        cursor.execute("SELECT major FROM users WHERE user_id = ?", (uid,))
        row = cursor.fetchone()

        if not row or not row[0]:
            user_states[uid] = "awaiting_major_for_games"
            bot.send_message(uid, "🧠 قبل أن نبدأ اللعب، أخبرنا بتخصصك:")
            return

        keyboard = InlineKeyboardMarkup(row_width=1)
        keyboard.add(
            InlineKeyboardButton("🔒 العب في الخاص", callback_data="game_private"),
            InlineKeyboardButton("👥 العب في المجموعة", switch_inline_query="game"),
            InlineKeyboardButton("🏠 القائمة الرئيسية", callback_data="go_back_home")
        )
        bot.edit_message_text(
            "🎮 اختر طريقة اللعب:\n\n"
            "- 🔒 في الخاص (ألعاب شخصية حسب تخصصك)\n"
            "- 👥 في المجموعة (شارك الأصدقاء بالتحدي!)",
            chat_id=chat_id,
            message_id=message_id,
            reply_markup=keyboard
        )
    
    elif data == "go_back_home":
        # إعادة عرض واجهة البداية
        keyboard = InlineKeyboardMarkup(row_width=2)
        buttons = [
            InlineKeyboardButton("📝 توليد اختبار", callback_data="go_generate"),
            InlineKeyboardButton("📚 مراجعة سريعة", callback_data="soon_review"),
            InlineKeyboardButton("📄 ملخص PDF", callback_data="soon_summary"),
            InlineKeyboardButton("🧠 بطاقات Anki", callback_data="anki"),
            InlineKeyboardButton("🎮 ألعاب تعليمية", callback_data="go_games"),
            InlineKeyboardButton("⚙️ حسابي", callback_data="soon_account"),
        ]
        keyboard.add(*buttons)

        bot.edit_message_text(
            "👋 أهلاً بك في TestGenie ✨\n\n"
            "🎯 أدوات تعليمية ذكية:\n"
            "- اختبارات من ملفاتك\n"
            "- بطاقات مراجعة (Anki)\n"
            "- ملخصات PDF/Word\n"
            "- ألعاب تعليمية *(قريبًا)*\n\n"
            "📌 لديك 3 اختبارات مجانية شهريًا.\n\n"
            "اختر ما يناسبك 👇",
            chat_id=chat_id,
            message_id=message_id,
            reply_markup=keyboard
        )


    elif data == "go_account_settings":
        bot.answer_callback_query(c.id)
        settings_keyboard = types.InlineKeyboardMarkup()
        settings_keyboard.add(
            InlineKeyboardButton("🎓 تغيير التخصص", callback_data="change_specialty"),
        )
        settings_keyboard.add(
            InlineKeyboardButton("⬅️ رجوع", callback_data="go_back_home")
        )

        bot.send_message(
            uid,
            "⚙️ *إعدادات الحساب*\n\n"
            "يمكنك تخصيص تجربتك التعليمية هنا.\n"
            "اختر ما ترغب بتعديله 👇",
            reply_markup=settings_keyboard,
            parse_mode="Markdown"
        )

    elif data == "change_specialty":
        keyboard = InlineKeyboardMarkup()
        buttons = [
            ("🩺 الطب", "major_الطب"),
            ("🛠️ الهندسة", "major_الهندسة"),
            ("💊 الصيدلة", "major_الصيدلة"),
            ("🗣️ اللغات", "major_اللغات"),
            ("❓ غير ذلك...", "major_custom"),
        ]
        for text, data_btn in buttons:
            keyboard.add(InlineKeyboardButton(text, callback_data=data_btn))
        keyboard.add(InlineKeyboardButton("⬅️ رجوع", callback_data="go_back_home"))

        bot.edit_message_text(
            "اختر تخصصك للبدء 👇", 
            chat_id=chat_id,
            message_id=message_id,
            reply_markup=keyboard
        )
        
    elif data.startswith("major_"):
        major_key = data.split("_", 1)[1]
        if major_key == "custom":
            user_states[uid] = "awaiting_major"
            bot.send_message(uid, "✏️ من فضلك أرسل اسم تخصصك بدقة.")
        else:
            cursor.execute("INSERT OR REPLACE INTO users(user_id, major) VALUES(?, ?)", (uid, major_key))
            conn.commit()
            bot.send_message(uid, f"✅ تم تحديد تخصصك: {major_key}\n"
                             "الآن أرسل ملف (PDF/DOCX/TXT) أو نصًا مباشرًا لتوليد اختبارك.")
    
    elif data == "game_private":
        try:
            cursor.execute("SELECT major FROM users WHERE user_id = ?", (uid,))
            row = cursor.fetchone()
            major = row[0] if row else "عام"

            keyboard = InlineKeyboardMarkup(row_width=1)
            keyboard.add(
                InlineKeyboardButton("🧩 Vocabulary Match", callback_data="game_vocab"),
                InlineKeyboardButton("⏱️ تحدي السرعة", callback_data="game_speed"),
                InlineKeyboardButton("❌ الأخطاء الشائعة", callback_data="game_mistakes"),
                InlineKeyboardButton("🧠 لعبة الاستنتاج", callback_data="game_inference"),
                InlineKeyboardButton("⬅️ رجوع", callback_data="go_games")
            )
            bot.edit_message_text(
                f"🎓 تخصصك الحالي: {major}\n"
                "اختر لعبة 👇",
                chat_id=chat_id,
                message_id=message_id,
                reply_markup=keyboard
            )
        except Exception as e:
            logging.exception("❌ حدث خطأ في game_private")
            bot.send_message(uid, "❌ حدث خطأ أثناء عرض الألعاب.")

    
    elif data == "back_to_games":
        try:
            bot.delete_message(c.message.chat.id, c.message.message_id)
        except Exception as e:
            logging.warning(f"❌ فشل حذف الرسالة عند الرجوع: {e}")
    
    

    elif data in ["game_vocab", "game_speed", "game_mistakes", "game_inference"]:
        game_type = data.split("_", 1)[1]

        # التحقق من إمكانية اللعب اليومي (6 مرات)
        state = game_states.get(uid, {"count": 0})
        if state["count"] >= 6:
            return bot.send_message(uid, "🛑 لقد وصلت إلى الحد الأقصى للألعاب المجانية (6 مرات).")

        if not can_play_game_today(uid, game_type):
            bot.answer_callback_query(c.id, "❌ لقد لعبت هذه اللعبة اليوم!")
            return

        loading_msg = bot.send_message(chat_id, "⏳ جاري تحضير السؤال...")

        try:
            record_game_attempt(uid, game_type)

            # التخصص
            cursor.execute("SELECT major FROM users WHERE user_id=?", (uid,))
            row = cursor.fetchone()
            major = row[0] if row else "عام"

            # توليد السؤال حسب نوع اللعبة
            if game_type == "vocab":
                raw = generate_vocabulary_game(uid, major, native_lang="Arabic")
            elif game_type == "speed":
                raw = generate_speed_challenge(uid, major, native_lang="Arabic")
            elif game_type == "mistakes":
                raw = generate_common_mistakes_game(uid, major, native_lang="Arabic")
            elif game_type == "inference":
                raw = generate_inference_game(uid, major, native_lang="Arabic")

            question = raw["question"]
            options = raw["options"]
            correct_index = raw["correct_index"]

            if not isinstance(options, list) or len(options) < 2:
                raise ValueError("عدد الخيارات غير صالح")

            # حفظ خيارات السؤال في الذاكرة المؤقتة
            game_states[uid] = {"count": state["count"] + 1, "options": options}

            keyboard = InlineKeyboardMarkup(row_width=2)

            # أزرار الإجابات
            for i, option in enumerate(options):
                short_option = (option[:50] + "...") if len(option) > 50 else option
                callback_data = f"ans_{game_type}_{i}_{correct_index}"
                keyboard.add(InlineKeyboardButton(short_option, callback_data=callback_data))
    
            # أزرار التحكم
            keyboard.row(
                InlineKeyboardButton("🔄 سؤال جديد", callback_data=f"new_{game_type}"),
                InlineKeyboardButton("⬅️ رجوع", callback_data="back_to_games")
            )
            keyboard.add(
                InlineKeyboardButton(
                    "📤 شارك هذه اللعبة", 
                    switch_inline_query="جرب هذه اللعبة الرائعة من @Oiuhelper_bot 🎯")
            )

            bot.delete_message(chat_id, loading_msg.message_id)
            text = f"🧠 اختر الإجابة الصحيحة:\n\n{question}"
            bot.send_message(chat_id, text, reply_markup=keyboard)

        except Exception as e:
            try:
                bot.delete_message(chat_id, loading_msg.message_id)
            except:
                pass
            logging.error(f"فشل توليد اللعبة: {str(e)}")
            bot.send_message(uid, "❌ حدث خطأ أثناء توليد اللعبة، حاول لاحقاً")

    # معالجة طلب سؤال جديد
    


    elif data.startswith("new_"):
        game_type = data.split("_", 1)[1]

        # تحقق من عدد المحاولات (كما في القسم الرئيسي)
        state = game_states.get(uid, {"count": 0})
        if state["count"] >= 6:
            msg = random.choice([
                "🚫 وصلت إلى الحد الأقصى لعدد الأسئلة اليوم!\n✨ جرب غدًا أو شارك البوت مع أصدقائك!",
                "❌ انتهت محاولات اليوم! يمكنك المحاولة مجددًا لاحقًا.",
                "🛑 لا مزيد من الأسئلة الآن. عد لاحقًا لتكمل رحلتك!"
            ])
            return bot.answer_callback_query(c.id, msg, show_alert=True)

        loading_msg = bot.send_message(c.message.chat.id, "⏳ جاري تحضير السؤال التالي...")

        try:
            # توليد السؤال الجديد
            cursor.execute("SELECT major FROM users WHERE user_id=?", (uid,))
            row = cursor.fetchone()
            major = row[0] if row else "عام"

            game_generators = {
                "vocab": generate_vocabulary_game,
                "speed": generate_speed_challenge,
                "mistakes": generate_common_mistakes_game,
                "inference": generate_inference_game
            }

            raw = game_generators[game_type](uid, major)
            question = raw["question"]
            options = raw["options"]
            correct_index = raw["correct_index"]

            if not isinstance(options, list) or len(options) < 2:
                raise ValueError("عدد الخيارات غير صالح")

            # حفظ خيارات السؤال الجديد
            game_states[uid]["count"] += 1
            game_states[uid]["options"] = options

            # إنشاء الأزرار
            keyboard = InlineKeyboardMarkup(row_width=2)
            for i, option in enumerate(options):
                short_option = (option[:50] + "...") if len(option) > 50 else option
                callback_data = f"ans_{game_type}_{i}_{correct_index}"
                keyboard.add(InlineKeyboardButton(short_option, callback_data=callback_data))

            keyboard.row(
                InlineKeyboardButton("🔄 سؤال جديد", callback_data=f"new_{game_type}"),
                InlineKeyboardButton("⬅️ رجوع", callback_data="back_to_games")
            )
            keyboard.add(
                InlineKeyboardButton(
                    "📤 شارك هذه اللعبة", 
                    switch_inline_query="جرب هذه اللعبة الرائعة من @Oiuhelper_bot 🎯")
            )

            # تعديل نفس الرسالة
            bot.edit_message_text(
                text=f"🧠 اختر الإجابة الصحيحة:\n\n{question}",
                chat_id=c.message.chat.id,
                message_id=c.message.message_id,
                reply_markup=keyboard
            )

        except Exception as e:
            logging.error(f"❌ فشل توليد سؤال جديد: {e}")
            bot.answer_callback_query(c.id, "❌ فشل توليد السؤال")

        finally:
            try:
                bot.delete_message(c.message.chat.id, loading_msg.message_id)
            except:
                pass

    elif data.startswith("ans_"):
        parts = data.split("_")
        game_type = parts[1]
        selected = int(parts[2])
        correct = int(parts[3])

        options = game_states.get(uid, {}).get("options", [])
        correct_text = options[correct] if correct < len(options) else f"الخيار رقم {correct+1}"

        wrong_responses = [
            "❌ خطأ! جرب مجددًا 😉\n✅ الصحيح: {correct}",
            "🚫 للأسف، ليست الصحيحة!\n✅ الجواب: {correct}",
            "😅 ليست الإجابة الصحيحة، الجواب هو: {correct}",
            "❌ لا، حاول مرة أخرى!\n✔️ الصحيح هو: {correct}"
        ]

        if selected == correct:
            bot.answer_callback_query(c.id, "✅ إجابة صحيحة!", show_alert=False)
        else:
            msg = random.choice(wrong_responses).format(correct=correct_text)
            bot.answer_callback_query(c.id, msg, show_alert=False)
    
    elif data.startswith("soon_"):
        feature_name = {
            "soon_review": "📚 ميزة المراجعة السريعة",
            "soon_summary": "📄 ملخصات PDF",
            "soon_account": "⚙️ إدارة الحساب",
        }.get(data, "هذه الميزة")

        bot.answer_callback_query(c.id)
        bot.send_message(chat_id, f"{feature_name} ستكون متاحة قريبًا... 🚧")



@bot.message_handler(func=lambda m: user_states.get(m.from_user.id) == "awaiting_major", content_types=['text'])
def set_custom_major(msg):
    if msg.chat.type != "private":
        return  # تجاهل الرسائل في المجموعات
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
 
@bot.message_handler(func=lambda m: user_states.get(m.from_user.id) in ["awaiting_major", "awaiting_major_for_games"])
def handle_user_major(msg):
    if msg.chat.type != "private":
        return  # تجاهل الرسائل في المجموعات
    uid = msg.from_user.id
    state = user_states.get(uid)
    major = msg.text.strip()

    cursor.execute("INSERT OR REPLACE INTO users(user_id, major) VALUES(?, ?)", (uid, major))
    conn.commit()
    user_states.pop(uid, None)

    if state == "awaiting_major":
        bot.send_message(uid, f"✅ تم تسجيل تخصصك: {major}\n"
                         "الآن أرسل ملف (PDF/DOCX/TXT) أو نصًا مباشرًا لتوليد اختبارك.")
    elif state == "awaiting_major_for_games":
        bot.send_message(uid, f"✅ تم تسجيل تخصصك: {major}\n"
                         "الآن يمكنك اختيار لعبة من قائمة الألعاب التعليمية.")
        # نرسل واجهة الألعاب مرة أخرى
        keyboard = InlineKeyboardMarkup(row_width=1)
        keyboard.add(
            InlineKeyboardButton("🔒 العب في الخاص", callback_data="game_private"),
            InlineKeyboardButton("👥 العب في المجموعة", switch_inline_query="game")
        )
        bot.send_message(uid, "🎮 اختر طريقة اللعب:", reply_markup=keyboard)




@bot.message_handler(content_types=['text', 'document'])
def unified_handler(msg):
    if msg.chat.type != "private":
        return
    
    uid = msg.from_user.id
    state = user_states.get(uid)

    # التخصص من قاعدة البيانات
    cursor.execute("SELECT major FROM users WHERE user_id = ?", (uid,))
    row = cursor.fetchone()
    major = row[0] if row else "General"

    # استخراج النص
    if msg.content_type == "text":
        content = msg.text[:3000]

    elif msg.content_type == "document":
        file_info = bot.get_file(msg.document.file_id)
        if file_info.file_size > 5 * 1024 * 1024:
            return bot.send_message(uid, "❌ الملف كبير جدًا، الحد 5 ميغابايت.")
    
        file_data = bot.download_file(file_info.file_path)
        os.makedirs("downloads", exist_ok=True)
        path = os.path.join("downloads", msg.document.file_name)

        with open(path, "wb") as f:
            f.write(file_data)

        ext = path.rsplit(".", 1)[-1].lower()
        if ext == "pdf":
            content = extract_text_from_pdf(path)[:3000]
        elif ext == "docx":
            content = extract_text_from_docx(path)[:3000]
        elif ext == "txt":
            content = extract_text_from_txt(path)[:3000]
        else:
            return bot.send_message(uid, "⚠️ نوع الملف غير مدعوم. أرسل PDF أو Word أو TXT.")

        try:
            os.remove(path)
        except Exception as e:
            print(f"[WARNING] لم يتم حذف الملف المؤقت: {e}")

        if not content or not content.strip():
            return bot.send_message(uid, "⚠️ لم أتمكن من قراءة محتوى الملف أو النص.")
        print(f">>> Content preview: {content[:300]}")


   
    # إذا المستخدم في وضع توليد أنكي
    if state == "awaiting_anki_file":
 
        user_states.pop(uid, None)
        bot.send_message(uid, "⏳ جاري إنشاء بطاقات المراجعة...")

        cards, title = generate_anki_cards_from_text(content, major=major, user_id=uid)

        if not cards:
            return bot.send_message(uid, "❌ لم أتمكن من توليد بطاقات.")

        # تنظيف العنوان ليكون اسم ملف صالح
        safe_title = re.sub(r'[^a-zA-Z0-9_\u0600-\u06FF]', '_', title)[:40]  # دعم الأسماء العربية وتحديد الطول

        filename = f"{safe_title}_{uid}.apkg"

        filepath = save_cards_to_apkg(cards, filename=filename, deck_name=title)
        bot.send_document(uid, open(filepath, 'rb'))



    # الحالة العادية: توليد اختبار
    else:
        if not can_generate(uid):
            return bot.send_message(uid, "⚠️ لقد استنفدت 3 اختبارات مجانية هذا الشهر.")
        
        bot.send_message(uid, "🧠 جاري توليد الاختبار، الرجاء الانتظار...")
        # ✅ هنا أضف الـ Debug قبل وبعد التوليد
        print(">>> Major:", major)
        print(">>> Content:", content[:300])
        quizzes = generate_quizzes_from_text(content, major, user_id=uid, num_quizzes=10)
        print(">>> Quizzes result:", quizzes)

        quizzes = generate_quizzes_from_text(content, major, user_id=uid, num_quizzes=10)

        if isinstance(quizzes, list) and len(quizzes) > 0:
            send_quizzes_as_polls(uid, quizzes)
            increment_count(uid)
        else:
            print("[ERROR] Failed to generate valid quizzes:", quizzes)
            bot.send_message(uid, "❌ فشل توليد الاختبار. حاول لاحقًا.")



# -------------------------------------------------------------------
#                   inference handler
# -------------------------------------------------------------------


@bot.message_handler(commands=['submit_inference'])
def handle_submit_inference(msg):
    if msg.chat.type != "private":
        return  # تجاهل الرسائل في المجموعات
    uid = msg.from_user.id
    user_states[uid] = {"state": "awaiting_inference_question", "temp": {}}
    bot.send_message(uid, "🧠 أرسل الآن سيناريو أو سؤالًا للاعبين (مثال: كيف تتصرف في هذا الموقف؟)")

@bot.message_handler(func=lambda m: user_states.get(m.from_user.id, {}).get("state") in [
    "awaiting_inference_question", "awaiting_inference_options", "awaiting_inference_correct"])
def handle_inference_submission(msg):
    if msg.chat.type != "private":
        return  # تجاهل الرسائل في المجموعات
    uid = msg.from_user.id
    state = user_states.get(uid, {})
    temp = state.get("temp", {})

    if state["state"] == "awaiting_inference_question":
        temp["question"] = msg.text.strip()
        user_states[uid] = {"state": "awaiting_inference_options", "temp": temp}
        bot.send_message(uid, "✏️ أرسل الآن 4 خيارات، كل خيار في سطر منفصل.")

    elif state["state"] == "awaiting_inference_options":
        options = msg.text.strip().split("\n")
        if len(options) != 4:
            return bot.send_message(uid, "⚠️ يجب أن ترسل 4 خيارات فقط، كل خيار في سطر.")
        temp["options"] = options
        user_states[uid] = {"state": "awaiting_inference_correct", "temp": temp}
        bot.send_message(uid, "✅ ما هو رقم الإجابة الصحيحة؟ (من 0 إلى 3)")

    elif state["state"] == "awaiting_inference_correct":
        try:
            correct = int(msg.text.strip())
            if correct not in [0, 1, 2, 3]:
                raise ValueError()
        except:
            return bot.send_message(uid, "⚠️ أرسل رقمًا صحيحًا من 0 إلى 3 فقط.")
        
        # حفظ في قاعدة البيانات
        q = temp["question"]
        options = temp["options"]
        options_str = json.dumps(options)
        cursor.execute("""
        INSERT INTO inference_questions (question, options, correct_index, submitted_by)
        VALUES (?, ?, ?, ?)
        """, (q, options_str, correct, uid))
        conn.commit()

        user_states.pop(uid, None)
        bot.send_message(uid, "🎉 تم استلام اقتراحك بنجاح! سيتم مراجعته قريبًا. شكراً لمساهمتك 🙏")
# -------------------------------------------------------------------
#                           Run Bot
# -------------------------------------------------------------------

# واجهة Flask للفحص
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ البوت يعمل الآن"
@app.route('/anki_preview')
def anki_preview():
    user_cards = generate_anki_cards_from_text(text)[:5]  # ← نحصل على أول 5 بطاقات
    session['cards'] = user_cards
    session['index'] = 0
    session['show_back'] = False
    return redirect('/anki')
    
app.secret_key = 'anki_secret'  # سر الجلسة لتخزين البيانات مؤقتًا


@app.route('/anki', methods=['GET', 'POST'])
def anki_cards():
    content = session.get('anki_content')
    major = session.get('anki_major', 'General')
    if 'cards' not in session:
        session['cards'] = example_cards[:5]
        session['index'] = 0
        session['show_back'] = False

    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'show':
            session['show_back'] = True
        elif action == 'next':
            session['index'] += 1
            session['show_back'] = False

    index = session['index']
    cards = session['cards']

    if index >= len(cards):
        session.clear()
        return "<h2>🎉 انتهيت من البطاقات! أحسنت.</h2><a href='/anki'>🔁 ابدأ من جديد</a>"

    return render_template('anki_viewer.html',
                           card=cards[index],
                           index=index,
                           total=len(cards),
                           show_back=session['show_back'])
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


