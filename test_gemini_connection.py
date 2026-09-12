"""
One-off sanity check: confirms the Gemini API key works and shows which
models are currently available to this key. Not part of the pipeline --
just a connectivity test, safe to delete afterward.
"""
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()  # reads GEMINI_API_KEY from .env

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("GEMINI_API_KEY not found -- check your .env file.")

client = genai.Client(api_key=api_key)

print("Models available to this key that support generateContent:\n")
for model in client.models.list():
    if "generateContent" in (model.supported_actions or []):
        print(" -", model.name)