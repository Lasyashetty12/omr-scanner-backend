import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Load environment variables from .env if present
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

# ============================================================
# SUPABASE & DATABASE CONFIGURATION
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    os.environ.get("SUPABASE_DB_URL")
)

if not DATABASE_URL:
    DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'omr_results.db')}"

# SQLAlchemy requires postgresql:// instead of postgres://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

TEMPLATE_DIR = os.path.join(
    BASE_DIR,
    "templates"
)

STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)

ANSWER_KEY_DIR = os.path.join(
    BASE_DIR,
    "answer_keys"
)

# Vercel writable temporary directories
UPLOAD_DIR = "/tmp/uploads"
RESULT_DIR = "/tmp/results"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# IMAGE QUALITY SETTINGS
# ============================================================

MIN_BLUR_SCORE = 80

MIN_BRIGHTNESS = 60

MAX_BRIGHTNESS = 245

MIN_CONTRAST = 20