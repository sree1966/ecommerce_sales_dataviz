import pandas as pd
import os

# --- Ensure you're in the project root directory ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "global_superstore_2016.csv")

# --- Load dataset with proper encoding ---
df = pd.read_csv(DATA_PATH, encoding='latin1')

# --- Fix date parsing (DD-MM-YYYY) ---
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True, errors='coerce')

# --- Add a 'Year' column for easier filtering later ---
df['Year'] = df['Order Date'].dt.year

# --- Drop rows with missing essential fields ---
df.dropna(subset=['Sales', 'Profit', 'Region', 'Category'], inplace=True)

# --- Save preprocessed dataset ---
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "preprocessed_data.csv")
df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

print(f"✅ Preprocessing complete! Saved cleaned data to:\n{OUTPUT_PATH}")
