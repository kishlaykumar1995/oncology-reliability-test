import pandas as pd
import numpy as np
import json
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# LOADING DATA

with open("extracted_oncology_data.json","r",encoding="utf-8") as f:
    data = json.load(f)

df = pd.json_normalize(data)
df = df.replace({None: np.nan, "": np.nan})

print("\n Loaded Data:", df.shape,"\n")
# SAMPLE
print(df[['filename','publication_year','predicted_reliability','design.study_type','design.sample_size']])
print("Rows:",len(df),"Columns:",df.shape[1],"\n")


# LABEL ENCODING (Target)

label_map = {"Low":0,"Medium":1,"High":2}
df["label"] = df["predicted_reliability"].map(label_map)


# STRUCTURED FEATURE ENGINEERING

# Study design indicators
df["is_rct"]    = df["design.study_type"].str.contains("RCT",case=False).astype(int)
df["is_meta"]   = df["design.study_type"].str.contains("Meta",case=False).astype(int)
df["is_observ"] = df["design.study_type"].str.contains("Retrospective|Cohort|Case",case=False).astype(int)

# Sample size tiers
df["large_study"]  = (df["design.sample_size"] > 500).astype(int)
df["medium_study"] = ((df["design.sample_size"]>100)&(df["design.sample_size"]<=500)).astype(int)
df["small_study"]  = (df["design.sample_size"]<=100).astype(int)

df["has_control"]  = df["design.comparator"].notna().astype(int)

# Follow-up extraction
def extract_months(text):
    if pd.isna(text): return np.nan
    text=text.lower()
    nums=re.findall(r"\d+",text)
    if "year" in text and nums: return int(nums[0])*12
    if "month" in text and nums: return int(nums[0])
    return np.nan

df["followup_months"] = df["design.followup_duration"].apply(extract_months)

# Endpoint type signals
df["primary_is_survival"] = df["design.primary_endpoint"].str.contains("OS|PFS|survival",case=False).astype(int)
df["primary_is_response"] = df["design.primary_endpoint"].str.contains("response|RR",case=False).astype(int)

# Quality indicators
df["reported_safety"]      = df["results.safety_profile_summary"].notna().astype(int)
df["reported_limitations"] = df["results.limitations_summary"].notna().astype(int)

df["years_since_2015"] = df["publication_year"].apply(lambda x:max(x-2015,0))

# Missing flags
missing_cols=["population.sex_distribution","population.comorbidities_summary","design.followup_duration"]
missing_flags = df[missing_cols].isna().astype(int)
missing_flags.columns=[c+"_missing" for c in missing_flags.columns]

df = pd.concat([df,missing_flags],axis=1)


# TF-IDF NLP FEATURE EXTRACTION

df["combined_text"] = (
    df["results.primary_result_text"].fillna("") + " " +
    df["results.safety_profile_summary"].fillna("") + " " +
    df["results.limitations_summary"].fillna("")
)

tfidf = TfidfVectorizer(max_features=300, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df["combined_text"])

print("TF-IDF text features:", tfidf_matrix.shape)
print("Top TF-IDF words:", tfidf.get_feature_names_out()[:15],"...")

def extract_first_number(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)
    return float(match.group(1)) if match else np.nan

df["hazard_ratio"] = df["combined_text"].apply(lambda t: extract_first_number(r"HR[:\s]*([0-9]*\.?[0-9]+)", t))
df["p_value"] = df["combined_text"].apply(lambda t: extract_first_number(r"p\s*=?\s*0?\.(\d+)", t))  # extracts decimals like 0.0074 → 74 →/10? If wanted exact, use full:
df["p_value_exact"] = df["combined_text"].apply(lambda t: extract_first_number(r"p\s*=?\s*(0?\.\d+)", t))

df["median_pfs_months"] = df["combined_text"].apply(lambda t: extract_first_number(r"median PFS(?:.*?)(\d+\.?\d*)", t))
df["median_os_months"] = df["combined_text"].apply(lambda t: extract_first_number(r"median OS(?:.*?)(\d+\.?\d*)", t))

df["survival_percent_improvement"] = df["combined_text"].apply(lambda t: extract_first_number(r"(\d+)% increase", t))
df["response_rate_percent"] = df["combined_text"].apply(lambda t: extract_first_number(r"(\d+)% response", t))

# Backup extraction of follow-up
df["followup_extracted_months"] = df["combined_text"].apply(
    lambda t: extract_first_number(r"(\d+)\s*month", t) 
              if pd.isna(extract_first_number(r"(\d+)\s*month", t)) == False 
              else np.nan
)

print("\nExtracted Clinical Outcome Features:")
print(df[[
    "hazard_ratio","p_value_exact","median_pfs_months",
    "median_os_months","survival_percent_improvement",
    "response_rate_percent","followup_extracted_months"
]])


# STRUCTURED NUMERIC FEATURE MATRIX

structured_features = [
    "design.sample_size","is_rct","is_meta","is_observ",
    "large_study","medium_study","small_study","has_control",
    "followup_months","primary_is_survival","primary_is_response",
    "reported_safety","reported_limitations","years_since_2015"
] + list(missing_flags.columns)

X_structured = df[structured_features].fillna(0).values

# Scale numeric values
scaler = StandardScaler()
X_struct_scaled = scaler.fit_transform(X_structured)

# Combine structured + text
from scipy.sparse import csr_matrix
X_final = hstack([csr_matrix(X_struct_scaled), tfidf_matrix])
y = df["label"].values

print("\n Final Feature Matrix Shape:", X_final.shape)


# SAVING EVERYTHING

df.to_csv("final_features.csv",index=False)
joblib.dump((X_final, y, tfidf, scaler),"feature_matrix.pkl")

print("\nSaved:")
print(" - final_features.csv")
print(" - feature_matrix.pkl  (contains features + label + vectorizer + scaler)\n")

print(" Feature Engineering Complete")
