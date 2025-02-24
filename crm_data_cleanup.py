import pandas as pd
import re
from fuzzywuzzy import fuzz, process
from sklearn.ensemble import IsolationForest
import requests

# Load CRM data with better error handling
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8', skipinitialspace=True)
        df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces
        print("Loaded data successfully with columns:", df.columns.tolist())
        return df
    except Exception as e:
        print("Error loading data:", e)
        return None

df = load_data("crm_data.csv")
if df is None:
    exit()

# Step 1: Data Cleansing & Standardization
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.strip().title()  # Remove spaces & capitalize
    text = re.sub(r'[^a-zA-Z0-9@. ]', '', text)  # Remove special chars
    return text

if 'Customer Name' in df.columns:
    df['Customer Name'] = df['Customer Name'].apply(clean_text)
if 'Email' in df.columns:
    df['Email'] = df['Email'].str.lower().str.strip()
if 'Phone' in df.columns:
    df['Phone'] = df['Phone'].astype(str).str.replace(r'\D', '', regex=True)  # Remove non-numeric chars

# Step 2: Duplicate Detection & Merging
def find_duplicates(name, choices, threshold=85):
    match, score = process.extractOne(name, choices)
    return match if score >= threshold else None

if 'Customer Name' in df.columns:
    df["Potential Duplicate"] = df["Customer Name"].apply(lambda x: find_duplicates(x, df["Customer Name"]))

# Step 3: Data Validation Using Anomaly Detection
if 'Phone' in df.columns:
    try:
        features = df[['Phone']].copy()
        features['Phone'] = features['Phone'].astype(float)
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        df['Anomaly Score'] = iso_forest.fit_predict(features[['Phone']])
        df['Valid Entry'] = df['Anomaly Score'].apply(lambda x: 'Valid' if x == 1 else 'Invalid')
    except Exception as e:
        print("Error in anomaly detection:", e)

# Step 4: Data Enrichment (Fetching Missing Data from External APIs)
def enrich_email(email):
    api_url = f"https://api.example.com/enrich?email={email}"  # Example API
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json().get("company", "Unknown")
    except Exception as e:
        print("Error fetching data for email:", email, e)
    return "Unknown"

if 'Email' in df.columns:
    df['Company'] = df['Email'].apply(enrich_email)

# Save cleaned data with better handling
def save_data(df, file_path):
    try:
        df.to_csv(file_path, index=False)
        print("Cleaned data saved successfully to", file_path)
    except Exception as e:
        print("Error saving data:", e)

save_data(df, "cleaned_crm_data.csv")

print("CRM Data Accuracy Improvement Completed!")
