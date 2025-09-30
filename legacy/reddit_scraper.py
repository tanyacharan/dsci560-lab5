import time
import re
import mysql.connector
from datetime import datetime, timezone, date
import requests
from io import BytesIO
from PIL import Image
import pytesseract
import praw
from collections import Counter
import spacy
from nltk.corpus import stopwords

# ---- CONFIG ----
SUBREDDITS = [
    "programming","technology","compsci","csMajors","learnprogramming",
    "datascience","MachineLearning","computers","hacking","coding",
    "cscareerquestions","InformationTechnology","ArtificialIntelligence",
    "technews","buildapc","linux","opensource","python","java","cpp"
]
NUM_POSTS = 5000

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "20020209",   # <- update if needed
    "database": "reddit_lab5"
}

# ---- PRAW CONFIG ----
reddit = praw.Reddit(
    client_id="TUNU3fm47cPMzof4ad73ZQ",
    client_secret="GkuE2L6gdFfi0QnY1S8Wjqo5Zf_7Ug",
    user_agent="lab5 scraper by u/florida_girl24"
)

# ---- NLP SETUP ----
nlp = spacy.load("en_core_web_sm", exclude=["ner"])  # lighter/faster; we don't need NER for keywords
SPACY_STOP = nlp.Defaults.stop_words
NLTK_STOP = set(stopwords.words("english"))
STOP_WORDS = (SPACY_STOP | NLTK_STOP)
VALID_POS = {"NOUN","PROPN","ADJ","VERB"}  # terms worth keeping

TAG_RULES = {
    "internship": ["internship","intern","return offer","summer 202", "fall 202", "winter 202", "spring 202"],
    "job": ["job","hiring","offer","new grad","recruiter","full-time","ft"],
    "rant": ["rant","vent","frustrated","annoyed","angry","toxic"],
    "interview": ["interview","oa","onsite","phone screen","behavioral","leetcode"],
    "resume": ["resume","cv","cover letter","portfolio","linkedin"],
    "help": ["help","advice","how do i","should i","any tips","stuck"]
}

# ---- DB SETUP ----
def init_db():
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    # Create table with new columns for keywords, tags, created_date
    cur.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id VARCHAR(20) PRIMARY KEY,
            title TEXT,
            author VARCHAR(50),
            url TEXT,
            score INT,
            num_comments INT,
            created_utc BIGINT,
            created_iso VARCHAR(50),
            created_date DATE,
            subreddit VARCHAR(50),
            selftext TEXT,
            ocr_text TEXT,
            keywords TEXT,
            tags VARCHAR(255)
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """)
    conn.commit()

    # If your MySQL < 8.0 and doesn't support IF NOT EXISTS on ADD COLUMN,
    # comment these out and add manually; otherwise they are harmless no-ops.
    try:
        cur.execute("ALTER TABLE posts ADD COLUMN IF NOT EXISTS keywords TEXT")
        cur.execute("ALTER TABLE posts ADD COLUMN IF NOT EXISTS tags VARCHAR(255)")
        cur.execute("ALTER TABLE posts ADD COLUMN IF NOT EXISTS created_date DATE")
        conn.commit()
    except mysql.connector.Error:
        pass
    return conn

# ---- HELPERS ----
HTML_TAG_RE = re.compile(r"<.*?>")
NON_PRINT_RE = re.compile(r"[^A-Za-z0-9\s\.\,\!\?\:\;\-\(\)\/\+]")  # allow a bit more punctuation

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("\x00"," ")
    text = NON_PRINT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def mask_username(user):
    if user in ("[deleted]", "AutoModerator", None):
        return user
    # simple mask; keeps just the first char
    return (user[0] + "***") if len(user) > 0 else "***"

def extract_text_from_image(url):
    """Download an image and run OCR."""
    if not url:
        return ""
    try:
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        text = pytesseract.image_to_string(img)
        return clean_text(text)
    except Exception:
        return ""

def get_image_url(submission):
    """Detect if submission has an image and return its URL."""
    try:
        if getattr(submission, "post_hint", None) == "image":
            return submission.url
        if hasattr(submission, "preview") and submission.preview and "images" in submission.preview:
            return submission.preview["images"][0]["source"]["url"]
        if submission.url and submission.url.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp",".tiff")):
            return submission.url
    except Exception:
        pass
    return None

def extract_keywords(text: str, topk: int = 12):
    """Very lightweight keyword extraction: lemmatize; keep NOUN/PROPN/ADJ/VERB; remove stopwords; top-k by frequency."""
    if not text:
        return []
    doc = nlp(text.lower())
    terms = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.like_num:
            continue
        if tok.pos_ in VALID_POS:
            lemma = tok.lemma_.strip()
            if len(lemma) >= 3 and lemma not in STOP_WORDS:
                terms.append(lemma)
    counts = Counter(terms)
    return [w for w,_ in counts.most_common(topk)]

def classify_tags(text: str):
    """Rule-based topical tags the lab cares about (internship, job, rant, etc.)."""
    t = (text or "").lower()
    found = set()
    for label, kws in TAG_RULES.items():
        for kw in kws:
            if kw in t:
                found.add(label)
                break
    return sorted(found)

def save_post(conn, submission, subreddit_name):
    cur = conn.cursor()

    # OCR text
    image_url = get_image_url(submission)
    ocr_text = extract_text_from_image(image_url) if image_url else ""

    title = clean_text(getattr(submission, "title", "") or "")
    body  = clean_text(getattr(submission, "selftext", "") or "")

    # build full text for NLP (title + body + OCR)
    full_text = " ".join([title, body, ocr_text]).strip()
    key_list = extract_keywords(full_text, topk=12)
    tags_list = classify_tags(full_text)

    # timestamps
    created_utc = int(getattr(submission, "created_utc", 0) or 0)
    created_iso = datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat() if created_utc else ""
    created_date = datetime.fromtimestamp(created_utc, tz=timezone.utc).date() if created_utc else None  # YYYY-MM-DD

    cur.execute("""
        INSERT IGNORE INTO posts
        (id, title, author, url, score, num_comments,
         created_utc, created_iso, created_date, subreddit, selftext, ocr_text, keywords, tags)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        submission.id,
        title,
        mask_username(submission.author.name if submission.author else None),
        submission.url,
        int(getattr(submission, "score", 0) or 0),
        int(getattr(submission, "num_comments", 0) or 0),
        created_utc,
        created_iso,
        created_date,                 # <- YYYY-MM-DD
        subreddit_name,
        body,
        ocr_text,
        ", ".join(key_list),          # <- keywords stored as CSV string
        ", ".join(tags_list)          # <- tags stored as CSV string
    ))
    conn.commit()

def scrape_subreddit(conn, subreddit_name, max_posts):
    subreddit = reddit.subreddit(subreddit_name)
    count = 0
    # NOTE: Reddit listing endpoints cap around 1k per listing.
    # Since you’re splitting across many subs, this still grows your dataset,
    # but for deep history from one sub you’ll want Pushshift + PRAW later.
    for submission in subreddit.new(limit=max_posts):
        save_post(conn, submission, subreddit_name)
        count += 1
        if count % 100 == 0:
            print(f"Fetched {count} posts so far from r/{subreddit_name}...")
    return count

def main():
    conn = init_db()
    total_fetched = 0
    for sub in SUBREDDITS:
        remaining = NUM_POSTS - total_fetched
        if remaining <= 0:
            break
        print(f"\nScraping r/{sub} (up to {remaining} posts left to reach {NUM_POSTS})...")
        fetched = scrape_subreddit(conn, sub, remaining)
        total_fetched += fetched
        print(f"Finished r/{sub}: {fetched} posts")
    print(f"\nDone. Total posts inserted: {total_fetched}")
    conn.close()

if __name__ == "__main__":
    main()
