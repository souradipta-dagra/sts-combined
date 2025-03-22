# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import pandas as pd
import regex as re
import unicodedata

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# List of dataset names
dataset_names = [
    "LMRC", "sts_chile_ns16", "sts_dubai", "sts_222_emr", "sts_india",
    "sts_italy", "sts_itac_nantes", "sts_kz8a", "sts_kz4at", "sts_rem",
    "sts_panama", "sts_net2", "sts_spain", "sts_reg2n", "sts_tib",
    "sts_xtrapolis_chile", "sts_vline_rrsmc", "sts_u400_Lyon", "sts_u400"
]

# Load all datasets
print("Loading Datasets... ")
dataframes = {name: dataiku.Dataset(name).get_dataframe() for name in dataset_names}
print("Datasets Loaded! ✅")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("Filtering incomplete and invalid records...")

# Define invalid patterns for 'observation' column
invalid_patterns = [r"####", r"#NAME?", r"-", r"^\d+$", r"^\s*$", r"^[!@#$%^&*(),.?\":{}|<>]+$", r"N/A - N/A"]

for name, df in dataframes.items():
    if "observation" in df.columns:
        # Remove rows where 'observation' contains invalid values
        df = df[~df["observation"].astype(str).str.match("|".join(invalid_patterns), na=False)]

    # Drop rows where either 'observation' or 'solution' is missing
    df = df.dropna(subset=["observation", "solution"], how="any")

    dataframes[name] = df

print("Data cleaning complete! ✅")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("Standardizing metadata...")

for name, df in dataframes.items():
    mode_values = {}

    for col in ["project", "database", "language"]:
        if col in df.columns and not df[col].dropna().empty:
            mode_values[col] = df[col].value_counts().idxmax()
        else:
            mode_values[col] = name  # Use dataset name as fallback

    # Ensure missing columns are added with the fallback value
    for col in ["project", "database", "language"]:
        df[col] = df[col].fillna(mode_values[col]) if col in df.columns else mode_values[col]

    dataframes[name] = df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("Checking unique values for key columns...")

for name, df in dataframes.items():
    unique_values = {col: df[col].nunique() if col in df.columns else 0 for col in ["project", "database", "language"]}

    if any(val > 1 for val in unique_values.values()):
        print(f"\n⚠️ {name}: Columns with multiple values -> { {k: v for k, v in unique_values.items() if v > 1} }")
        for col in ["project", "database", "language"]:
            if col in df.columns and df[col].nunique() > 1:
                print(f"🔍 Unique values in '{col}' for {name}: {df[col].unique()}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("Standardizing inconsistent values...")

for name, df in dataframes.items():
    if name == "LMRC" and "language" in df.columns:
        df["language"] = df["language"].replace({"ENGLISH": "English"})

    if name == "sts_222_emr" and "project" in df.columns:
        df["project"] = "222 - EMR"

    if name == "sts_xtrapolis_chile" and "project" in df.columns:
        df["project"] = df["project"].replace({"MERVAL": "Merval", "merval": "Merval"})

    if name == "sts_u400":
        if "language" in df.columns:
            df["language"] = df["language"].replace({"ENGLISH": "English", "SPANISH": "Spanish"})
    if "database" in df.columns:
        df = df[df["database"] != "STS_U400_6.0"]

    dataframes[name] = df

print("Standardization complete! ✅")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("Checking unique values for datasets with multiple values in key columns...")

for name, df in dataframes.items():
    unique_values = {col: df[col].nunique() if col in df.columns else 0 for col in ["project", "database", "language"]}

    if any(val > 1 for val in unique_values.values()):  # If any column has more than one unique value
        print(f"\n⚠️ {name}: Columns with multiple values -> { {k: v for k, v in unique_values.items() if v > 1} }")
        for col in ["project", "database", "language"]:
            if col in df.columns and df[col].nunique() > 1:
                print(f"🔍 Unique values in '{col}' for {name}: {df[col].unique()}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def clean_text(text):
    if pd.isna(text):
        return ""

    # Convert to string and normalize unicode characters (fixes encodings)
    text = unicodedata.normalize('NFKC', str(text)).strip()

    # Fix spaces around abbreviations
    text = re.sub(r'(?<=[a-zA-Z])\.(?=[a-zA-Z])', '. ', text)

    # Keep decimal numbers intact
    text = re.sub(r'(?<=\d)\s*\.\s*(?=\d)', '.', text)

    # Normalize multiple spaces, tabs, or newlines
    text = re.sub(r'\s+', ' ', text)

    # Remove unwanted special characters **but keep** technical symbols
    text = re.sub(r"[^a-zA-Z0-9\s\-.,;:!?()/\\%#@$*+=\"'_<>]", "", text)

    return text.strip()

# Preprocess all datasets
print("Applying preprocessing...")
for name, df in dataframes.items():
    print(f"Preprocessing {name}...")
    if "observation" in df.columns and "problemcause" in df.columns and "solution" in df.columns:
        df[["observation","problemcause","solution"]] = df[["observation","problemcause","solution"]].map(clean_text)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("Merging all dataframes into a single dataset...")

# Combine all DataFrames
combined_df = pd.concat(dataframes.values(), ignore_index=True)
# combined_df = combined_df[combined_df["database"] != "ENGINEERING_7702"]


print(f"Merging complete! ✅ Final dataset has {combined_df.shape[0]} rows and {combined_df.shape[1]} columns.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(combined_df["language"].unique())
print(f"Total unique values in 'language': {combined_df['language'].nunique()}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
language_mapping = {
    "ENGLISH": "English",  # Normalize case
    "RUS": "Russian",  # Expand 'RUS' to 'Russian'
    "kazakh": "Kazakh",  # Normalize case
    "SWEDISH": "Swedish"  # Normalize case
}
if "language" in combined_df.columns:
    combined_df["language"] = combined_df["language"].replace(language_mapping)

# Define valid language mappings
valid_languages = {
    'english': 'en',
    'french': 'fr',
    'italian': 'it',
    'kazakh': 'kk',
    'russian': 'ru',
    'spanish': 'es',
    'swedish': 'sv'
}

# Function to standardize languages
def standardize_language(lang):
    if pd.isna(lang):
        return 'unknown'
    lang = str(lang).strip().lower()
    if lang not in valid_languages:
        print(f"⚠️ Unknown language detected: {lang}")
    return valid_languages.get(lang, 'unknown')

# Apply standardization
if "language" in combined_df.columns:
    combined_df["language"] = combined_df["language"].apply(standardize_language)

# Verify the changes
print("\n🔍 Unique values in 'language' after standardization:")
print(combined_df["language"].unique())
print(f"Total unique values in 'language': {combined_df['language'].nunique()}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("Standardizing 'database' values...")

database_mapping = {"Rex": "REX"}
if "database" in combined_df.columns:
    combined_df["database"] = combined_df["database"].replace(database_mapping)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("Checking unique values in 'project', 'database', and 'language' columns...\n")

for col in ["project", "database", "language"]:
    if col in combined_df.columns:
        unique_values = combined_df[col].unique()
        print(f"🔍 Unique values in '{col}':")
        print(unique_values)  # Display the unique values
        print(f"Total unique values in '{col}': {len(unique_values)}\n")
    else:
        print(f"⚠️ Column '{col}' not found in merged dataset.\n")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
combined_df.shape[0]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("Ensuring metadata columns exist...")

METADATA_COLUMNS = [
    'project', 'fleet', 'subsystem', 'database', 'observationcategory', 'problemcode',
    'problemcause', 'solutioncategory', 'language', 'failureclass', 'date', 'observation', 'solution'
]

for col in METADATA_COLUMNS:
    if col not in combined_df.columns:
        combined_df[col] = 'Unknown'

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
combined_df.shape[0]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Remove empty text entries
combined_df = combined_df[combined_df['observation'] != '']
combined_df.shape[0]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Identify problematic columns
mixed_type_cols = [col for col in combined_df.columns if combined_df[col].map(type).nunique() > 1]

# Convert only these columns to strings
for col in mixed_type_cols:
    combined_df[col] = combined_df[col].astype(str)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Shall drop many entries if you do so!
# combined_df = combined_df.drop_duplicates(subset=['project', 'database', 'observation', 'solution', 'language']).reset_index(drop=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("Checking unique values in 'project', 'database', and 'language' columns...\n")

for col in ["project", "database", "language"]:
    if col in combined_df.columns:
        unique_values = combined_df[col].unique()
        print(f"🔍 Unique values in '{col}':")
        print(unique_values)  # Display the unique values
        print(f"Total unique values in '{col}': {len(unique_values)}\n")
    else:
        print(f"⚠️ Column '{col}' not found in merged dataset.\n")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write to Dataiku
print("Saving enhanced knowledge base...")
output_file = 'sts_combined_data'
output_dataset = dataiku.Dataset(output_file)
output_dataset.write_with_schema(combined_df)
print(f"Enhanced knowledge base saved to '{output_file}'")