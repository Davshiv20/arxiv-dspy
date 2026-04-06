import dspy
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from langfuse import get_client

load_dotenv()

OR_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

langfuse = get_client()

lm = dspy.LM("openrouter/google/gemini-2.5-flash", api_key=OR_API_KEY)
dspy.configure(lm=lm)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def test_db():
    """Quick smoke test for Supabase connection and pgvector setup."""
    try:
        result = supabase.table("papers").select("id").limit(1).execute()
        print(f"DB connection OK. Papers table accessible. Rows: {len(result.data)}")
        return True
    except Exception as e:
        print(f"DB connection FAILED: {e}")
        return False


if __name__ == "__main__":
    test_db()
