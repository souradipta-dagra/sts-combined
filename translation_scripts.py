# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import pandas as pd
import datetime
import concurrent.futures
from tqdm.notebook import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read recipe inputs
sts_cmb = dataiku.Dataset("sts_cmb")
sts_cmb_df = sts_cmb.get_dataframe(infer_with_pandas=False)
sts_cmb_df['status'] = "New"

# Sample function - comment out for full processing
# def sample_projects(df, project_column='project', n_samples=10):
#     return df.groupby(project_column).apply(lambda x: x.sample(n=n_samples)).reset_index(drop=True)
# sts_cmb_df = sample_projects(sts_cmb_df, project_column='project', n_samples=20)

# Define the LLM ID
LLM_ID = "openai:Lite_llm_STS_Dev_GPT_4O:gpt-35-turbo-16k"

# Create a handle for the LLM
client = dataiku.api_client()
project = client.get_default_project()
llm = project.get_llm(LLM_ID)

# Dictionary of language codes to full names
language_map = {
    'en': 'English',
    'fr': 'French',
    'it': 'Italian',
    'kk': 'Kazakh',
    'ru': 'Russian',
    'es': 'Spanish',
    'sv': 'Swedish'
}

# Constants for chunking and processing
BATCH_SIZE = 100  # Process records in batches of 100
MAX_WORKERS = 4   # Number of parallel workers

# Function to translate a single record
def translate_record(record_data):
    index, row, language = record_data
    translations = {}
    lang_name = language_map.get(language, "Unknown")

    for column in ["observation", "problem_cause", "problem_code", "solution"]:
        # Skip empty fields for problem code and cause
        if column in ["problem_code", "problem_cause"] and pd.isna(row[column]):
            translations[f"{column}_translated"] = ''
            continue

        # Skip translation for English
        if language == 'en':
            translations[f"{column}_translated"] = row[column]
            continue

        message_text = f"""
        You are an expert language translator. Your task is to precisely translate the given text from {lang_name} to English. Please adhere to the following guidelines:
        1. Deliver only the translation without any additional commentary or explanations. Do not preamble.
        2. Ensure the translation is accurate and avoid generating any false or fabricated information. Clean the data by removing any Unicode and special characters.
        3. If the input text is empty, respond with an empty string.
        4. Do not add any punctuation.
        5. Ensure that the translated text maintains the meaning and context of the original text.

        Translate the following text: '{row[column]}'
        """

        try:
            completion = llm.new_completion()
            completion.with_message(message_text)
            resp = completion.execute()

            if resp.success:
                translations[f"{column}_translated"] = resp.text
            else:
                logging.error(f"Translation failed for index {index}, column {column}, language {lang_name}. Response: {resp.text}")
                translations[f"{column}_translated"] = row[column]  # Keep original on failure
        except Exception as e:
            logging.error(f"Error during translation for index {index}, column {column}: {e}")
            translations[f"{column}_translated"] = row[column]  # Keep original on failure

    return index, translations

# Function to process records in batches with parallelization
def process_in_batches(df, batch_size=BATCH_SIZE):
    results = {}
    new_records = df[df["status"] == "New"]
    total_records = len(new_records)

    # Create a list of all record data for processing
    all_records = [(idx, row, row['language']) for idx, row in new_records.iterrows()]

    # Process in batches
    for i in range(0, total_records, batch_size):
        batch = all_records[i:i+batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}/{(total_records + batch_size - 1)//batch_size}")

        # Process batch with parallel workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_record = {executor.submit(translate_record, record): record for record in batch}

            # Show progress bar for current batch
            for future in tqdm(concurrent.futures.as_completed(future_to_record),
                              total=len(batch),
                              desc=f"Batch {i//batch_size + 1}"):
                idx, translations = future.result()
                results[idx] = translations

    return results

# Main execution
start = datetime.datetime.now()

# Process the data in batches with parallelization
translation_results = process_in_batches(sts_cmb_df)

# Update the main dataframe with translation results
for idx, translations in translation_results.items():
    for col, value in translations.items():
        sts_cmb_df.at[idx, col] = value

# Update status for processed records
sts_cmb_df.loc[sts_cmb_df["status"] == "New", "status"] = "Processed"

end = datetime.datetime.now()
logging.info(f"Total processing time: {end - start}")

# Write recipe outputs
sts_cmb_trns = dataiku.Dataset("sts_cmb_trns")
sts_cmb_trns.write_with_schema(sts_cmb_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import pandas as pd
import datetime
import concurrent.futures
from tqdm.notebook import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the LLM ID
LLM_ID = "openai:Lite_llm_STS_Dev_GPT_4O:gpt-35-turbo-16k"

# Create a handle for the LLM
client = dataiku.api_client()
project = client.get_default_project()
llm = project.get_llm(LLM_ID)

# Dictionary of language codes to full names
language_map = {
    'en': 'English',
    'fr': 'French',
    'it': 'Italian',
    'kk': 'Kazakh',
    'ru': 'Russian',
    'es': 'Spanish',
    'sv': 'Swedish'
}

# Constants for chunking and processing
BATCH_SIZE = 100  # Process records in batches of 100
MAX_WORKERS = 4   # Number of parallel workers

# Function to translate a single record
def translate_record(record_data):
    index, row, language = record_data
    translations = {}
    lang_name = language_map.get(language, "Unknown")

    for column in ["observation", "problem_cause", "problem_code", "solution"]:
        # Skip empty fields for problem code and cause
        if column in ["problem_code", "problem_cause"] and pd.isna(row[column]):
            translations[f"{column}_translated"] = ''
            continue

        # Skip translation for English
        if language == 'en':
            translations[f"{column}_translated"] = row[column]
            continue

        message_text = f"""
        You are an expert language translator. Your task is to precisely translate the given text from {lang_name} to English. Please adhere to the following guidelines:
        1. Deliver only the translation without any additional commentary or explanations. Do not preamble.
        2. Ensure the translation is accurate and avoid generating any false or fabricated information. Clean the data by removing any Unicode and special characters.
        3. If the input text is empty, respond with an empty string.
        4. Do not add any punctuation.
        5. Ensure that the translated text maintains the meaning and context of the original text.

        Translate the following text: '{row[column]}'
        """

        try:
            completion = llm.new_completion()
            completion.with_message(message_text)
            resp = completion.execute()

            if resp.success:
                translations[f"{column}_translated"] = resp.text
            else:
                logging.error(f"Translation failed for index {index}, column {column}, language {lang_name}. Response: {resp.text}")
                translations[f"{column}_translated"] = row[column]  # Keep original on failure
        except Exception as e:
            logging.error(f"Error during translation for index {index}, column {column}: {e}")
            translations[f"{column}_translated"] = row[column]  # Keep original on failure

    return index, translations

# Function to process records in batches with parallelization
def process_in_batches(df, batch_size=BATCH_SIZE):
    results = {}
    new_records = df[df["status"] == "New"]
    total_records = len(new_records)

    # Create a list of all record data for processing
    all_records = [(idx, row, row['language']) for idx, row in new_records.iterrows()]

    # Process in batches
    for i in range(0, total_records, batch_size):
        batch = all_records[i:i+batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}/{(total_records + batch_size - 1)//batch_size}")

        # Process batch with parallel workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_record = {executor.submit(translate_record, record): record for record in batch}

            # Show progress bar for current batch
            for future in tqdm(concurrent.futures.as_completed(future_to_record),
                              total=len(batch),
                              desc=f"Batch {i//batch_size + 1}"):
                idx, translations = future.result()
                results[idx] = translations

    return results

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
sts_cmb_trns = dataiku.Dataset("sts_cmb_trns")
sts_cmb_trns_df = sts_cmb_trns.get_dataframe(infer_with_pandas=False)

for column in sts_cmb_trns_df.columns:
    if sts_cmb_trns_df[column].dtype == object:
        sts_cmb_trns_df[column] = sts_cmb_trns_df[column].fillna('')
    else:
        sts_cmb_trns_df[column] = sts_cmb_trns_df[column].fillna(0)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Define the columns to check for translation failures
columns_to_check = ["observation", "solution"]

def identify_failed_rows(df):
    # Filter for non-English rows
    non_english_rows = df[df['language'] != 'en']

    # Initialize a list to store indices of failed rows
    failed_indices = []

    # Check for matching original and translated values
    for idx, row in non_english_rows.iterrows():
        failed = False
        for col in columns_to_check:
            original_value = row[col]
            translated_value = row[f"{col}_translated"]

            # Skip checks if the original value is NaN
            if pd.isna(original_value):
                continue

            # Check if the translated value matches the original value
            if str(original_value).strip() == str(translated_value).strip():
                failed = True
                break  # No need to check further columns for this row

        if failed:
            failed_indices.append(idx)

    return failed_indices

# Identify failed rows
failed_indices = identify_failed_rows(sts_cmb_trns_df)

# Separate failed rows from the main dataset
failed_rows_df = sts_cmb_trns_df.loc[failed_indices].copy()
successful_rows_df = sts_cmb_trns_df.loc[~sts_cmb_trns_df.index.isin(failed_indices)].copy()

# Print summary
print(f"Total failed rows identified: {len(failed_rows_df)}")
print(f"Total successful rows: {len(successful_rows_df)}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
failed_rows_df['status'] = "New"

retry_results = process_in_batches(failed_rows_df, batch_size=100)

# Update the main DataFrame with retry results
for idx, translations in retry_results.items():
    for col, value in translations.items():
        sts_cmb_trns_df.at[idx, col] = value

# Mark retried rows as "Processed"
sts_cmb_trns_df.loc[failed_rows_df.index, 'status'] = "Processed"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Identify remaining failed rows after retry
remaining_failed_indices = identify_failed_rows(sts_cmb_trns_df)

# Separate remaining failed rows
remaining_failed_rows_df = sts_cmb_trns_df.loc[remaining_failed_indices].copy()

# Log summary
print(f"Remaining failed rows after retry: {len(remaining_failed_rows_df)}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
sts_cmb_trns_out = dataiku.Dataset("sts_cmb_trns_final")
sts_cmb_trns_out.write_with_schema(sts_cmb_trns_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import pandas as pd
import concurrent.futures
import logging
from tqdm.notebook import tqdm

# ------------------------- CONFIGURATION -------------------------
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the LLM ID
LLM_ID = "openai:Lite_llm_STS_Dev_GPT_4O:gpt-35-turbo-16k"

# Create a handle for the LLM
client = dataiku.api_client()
project = client.get_default_project()
llm = project.get_llm(LLM_ID)

# ------------------------- HELPER FUNCTIONS -------------------------
def is_blank(value):
    """Check if a value is blank or missing."""
    return pd.isna(value) or str(value).strip() == ''

def find_incomplete_translations(df, original_col, translated_col):
    """Identify rows with incomplete translations."""
    return df[original_col].notna() & ~df[original_col].apply(is_blank) & df[translated_col].apply(is_blank)

def translate_with_llm(record_data):
    """Translate a single record using the LLM."""
    index, row, language = record_data
    translations = {}
    language_map = {
        'en': 'English',
        'fr': 'French',
        'it': 'Italian',
        'kk': 'Kazakh',
        'ru': 'Russian',
        'es': 'Spanish',
        'sv': 'Swedish'
    }
    lang_name = language_map.get(language, "Unknown")

    for column in ["observation", "problem_cause", "problem_code", "solution"]:
        # Skip empty fields for problem code and cause
        if column in ["problem_code", "problem_cause"] and pd.isna(row[column]):
            translations[f"{column}_translated"] = row[column]
            continue

        # Skip translation for English
        if language == 'en':
            translations[f"{column}_translated"] = row[column]
            continue

        # LLM prompt for translation
        message_text = f"""
        You are an expert language translator. Your task is to precisely translate the given text from {lang_name} to English. Please adhere to the following guidelines:
        1. Deliver only the translation without any additional commentary or explanations. Do not preamble.
        2. Ensure the translation is accurate and avoid generating any false or fabricated information. Clean the data by removing any Unicode and special characters.
        3. If the input text is empty, respond with an empty string.
        4. Do not add any punctuation.
        5. Ensure that the translated text maintains the meaning and context of the original text.
        6. There should be no empty translation for original columns that have values. If translation is not possible, keep the original value.

        Translate the following text: '{row[column]}'
        """

        try:
            completion = llm.new_completion()
            completion.with_message(message_text)
            resp = completion.execute()

            if resp.success:
                translations[f"{column}_translated"] = resp.text
            else:
                logging.error(f"Translation failed for index {index}, column {column}, language {lang_name}. Response: {resp.text}")
                translations[f"{column}_translated"] = row[column]  # Keep original on failure
        except Exception as e:
            logging.error(f"Error during translation for index {index}, column {column}: {e}")
            translations[f"{column}_translated"] = row[column]  # Keep original on failure

    return index, translations

def process_in_batches(df, batch_size=100, max_workers=4):
    """Process records in batches with parallelization."""
    results = {}
    total_records = len(df)

    # Create a list of all record data for processing
    all_records = [(idx, row, row['language']) for idx, row in df.iterrows()]

    # Process in batches
    for i in range(0, total_records, batch_size):
        batch = all_records[i:i+batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}/{(total_records + batch_size - 1)//batch_size}")

        # Process batch with parallel workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_record = {executor.submit(translate_with_llm, record): record for record in batch}

            # Show progress bar for current batch
            for future in tqdm(concurrent.futures.as_completed(future_to_record),
                              total=len(batch),
                              desc=f"Batch {i//batch_size + 1}"):
                idx, translations = future.result()
                results[idx] = translations

    return results

# ------------------------- MAIN PIPELINE -------------------------
def translation_pipeline():
    """Main pipeline to handle translations."""
    # Step 1: Read recipe inputs
    sts_cmb_trns_final = dataiku.Dataset("sts_cmb_trns_final")
    translated_df = sts_cmb_trns_final.get_dataframe(infer_with_pandas=False)

    # Step 2: Define column pairs for original and translated columns
    column_pairs = [
        ("observation", "observation_translated"),
        ("solution", "solution_translated"),
        ("problem_cause", "problem_cause_translated"),
        ("problem_code", "problem_code_translated"),
    ]

    # Step 3: Identify rows with incomplete translations
    incomplete_conditions = [
        find_incomplete_translations(translated_df, orig_col, trans_col)
        for orig_col, trans_col in column_pairs
    ]
    combined_condition = incomplete_conditions[0]
    for condition in incomplete_conditions[1:]:
        combined_condition |= condition

    incomplete_rows_df = translated_df[combined_condition].copy()
    logging.info(f"Total rows with incomplete translations: {len(incomplete_rows_df)}")

    # Step 4: Translate incomplete rows
    if not incomplete_rows_df.empty:
        incomplete_rows_df['status'] = "New"
        retry_results = process_in_batches(incomplete_rows_df)

        # Update the main DataFrame with retry results
        for idx, translations in retry_results.items():
            for col, value in translations.items():
                translated_df.at[idx, col] = value

        # Mark retried rows as "Processed"
        translated_df.loc[incomplete_rows_df.index, 'status'] = "Processed"

    # Step 5: Ensure no blanks remain
    for orig_col, trans_col in column_pairs:
        translated_df.loc[
            find_incomplete_translations(translated_df, orig_col, trans_col),
            trans_col
        ] = translated_df[orig_col]

    # Step 6: Additional processing
    translated_df['combined_translated'] = translated_df.apply(
        lambda row: f"Observation: {row['observation_translated'] if pd.notna(row['observation_translated']) else ''} [SEP] Solution: {row['solution_translated'] if pd.notna(row['solution_translated']) else ''} [SEP] Problem Cause: {row['problem_cause_translated'] if pd.notna(row['problem_cause_translated']) else ''} [SEP] Problem Code: {row['problem_code_translated'] if pd.notna(row['problem_code_translated']) else ''}",
        axis=1
    )

    # Step 7: Write recipe outputs
    sts_cmb_trns_final_filled = dataiku.Dataset("sts_cmb_trns_final_filled")
    sts_cmb_trns_final_filled.write_with_schema(translated_df)
    logging.info("Updated DataFrame written to sts_cmb_trns_final_filled.")

# Execute the pipeline
translation_pipeline()
