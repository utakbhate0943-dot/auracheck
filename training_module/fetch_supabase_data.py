"""
Supabase data fetching module for populating training datasets.

Retrieves student mental health data from Supabase cloud database and exports
to local CSV for model training. Used at application startup to ensure training
data is current with live database.

Functions:
    fetch_students_mental_health: Download and save students_mental_health table to CSV
"""

import os
import pandas as pd
from typing import Optional
from supabase import create_client


def fetch_students_mental_health() -> Optional[str]:
    """
    Fetch students_mental_health data from Supabase and save to local CSV.
    
    **Data Flow:**
    1. Reads SUPABASE_URL and SUPABASE_ANON_KEY from environment
    2. Queries `students_mental_health` table via Supabase REST API
    3. Converts result to pandas DataFrame
    4. Saves to training_output/students_mental_health.csv
    
    This CSV is then used for:
    - Training Logistic Regression, Gradient Boosting, KMeans models
    - Feature engineering (wellbeing target calculation)
    - Preprocessing pipeline fitting
    
    Returns:
        Optional[str]: Full path to saved CSV file if successful, None if table empty or error
    
    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_ANON_KEY not set in environment
        Exception: If network error or Supabase query fails (logged as warning)
    
    Example:
        >>> csv_path = fetch_students_mental_health()
        >>> if csv_path:
        ...     df = pd.read_csv(csv_path)
        ...     print(f"Loaded {len(df)} records from {csv_path}")
    
    **Environment Setup:**
        Set in .env file:
        SUPABASE_URL=https://your-project.supabase.co
        SUPABASE_ANON_KEY=your_supabase_anon_key_here
    """
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
