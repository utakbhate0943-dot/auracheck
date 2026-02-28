import os
import pandas as pd
from supabase import create_client

def fetch_students_mental_health():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise ValueError("Supabase credentials not set in environment.")
    supabase = create_client(url, key)
    data = supabase.table("students_mental_health").select("*").execute()
    df = pd.DataFrame(data.data)
    output_path = os.path.join(os.path.dirname(__file__), "..", "training_output", "students_mental_health.csv")
    if df.empty:
        print("Warning: No data found in students_mental_health table. CSV not written.")
        return None
    df.to_csv(output_path, index=False)
    return output_path