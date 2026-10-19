import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DB_PATH = Path(os.getenv("DUM_DB_PATH", BASE_DIR / "data" / "dum_agent.db"))
REPORTS_DIR = Path(os.getenv("DUM_REPORTS_DIR", BASE_DIR / "reports"))

# Azure OpenAI
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # your deployment NAME
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_TEMPERATURE = os.getenv("AZURE_OPENAI_TEMPERATURE")  # optional

# Keyword shortcuts for intent routing (saves one LLM call per message)
FAST_ROUTING = os.getenv("DUM_FAST_ROUTING", "true").lower() in {"1", "true", "yes"}

# After this many follow-up rounds without a usable update, stop asking and drop the draft
MAX_CLARIFY_ROUNDS = int(os.getenv("DUM_MAX_CLARIFY_ROUNDS", "2"))

CANCEL_WORDS = {"cancel", "skip", "never mind", "nevermind", "discard", "forget it"}
