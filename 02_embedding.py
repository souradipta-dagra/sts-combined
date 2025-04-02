import dataiku
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle
import pandas as pd
from tqdm import tqdm
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname=s) - %(message=s)')

# Load the dataset
logging.info("Loading the dataset...")
knowledge_base = dataiku.Dataset("combined_data").get_dataframe(infer_with_pandas=False)

# Identify and convert problematic columns to strings
logging.info("Identifying and converting problematic columns to strings...")
mixed_type_cols = [col for col in knowledge_base.columns if knowledge_base[col].map(type).nunique() > 1]
for col in mixed_type_cols:
    knowledge_base[col] = knowledge_base[col].astype(str)

# Load Sentence Transformer model
logging.info("Loading Sentence Transformer model...")
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Function to generate and store separate embeddings
def generate_and_store_embeddings(column_name, index_name):
    logging.info(f"Generating embeddings for {column_name}...")
    embeddings = embedding_model.encode(
        knowledge_base[column_name].tolist(), show_progress_bar=True, batch_size=32
    )
    embeddings = np.array(embeddings).astype('float32')  # FAISS expects float32

    # Normalize embeddings for better FAISS search
    faiss.normalize_L2(embeddings)

    # Build a FAISS HNSW index for dense retrieval
    logging.info(f"Building a FAISS HNSW index for {column_name}...")
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 40 
    index.add(embeddings)
    logging.info(f"FAISS HNSW index for {column_name} created with {index.ntotal} entries.")

    # Save FAISS index to temp
    index_path = f"/tmp/{index_name}.faiss"
    faiss.write_index(index, index_path)
    return index_path

# Generate and store embeddings for each column
obs_index_path = generate_and_store_embeddings('observation', 'obs_transformers')
problem_cause_index_path = generate_and_store_embeddings('problem_cause', 'problem_cause_transformers')
problem_code_index_path = generate_and_store_embeddings('problem_code', 'problem_code_transformers')
solution_index_path = generate_and_store_embeddings('solution', 'solution_transformers')

# Generate combined text for unified embedding
logging.info("Generating combined text for unified embedding...")
knowledge_base['combined_text'] = ["observation: " + obs + " problem_cause: " + pc + " problem_code: " + pcode + " solution: " + sol
                          for obs, pc, pcode, sol in zip(knowledge_base['observation'], knowledge_base['problem_cause'], knowledge_base['problem_code'], knowledge_base['solution'])]

# Generate single-vector embeddings for the combined text
logging.info("Generating single-vector embeddings for combined text...")
combined_embeddings = embedding_model.encode(
    knowledge_base['combined_text'].tolist(), show_progress_bar=True, batch_size=32
)
combined_embeddings = np.array(combined_embeddings).astype('float32')  # FAISS expects float32

# Normalize embeddings for better FAISS search
faiss.normalize_L2(combined_embeddings)

# Build a FAISS HNSW index for dense retrieval
logging.info("Building a FAISS HNSW index for combined text...")
dimension = combined_embeddings.shape[1]
combined_index = faiss.IndexHNSWFlat(dimension, 32)
combined_index.hnsw.efConstruction = 40 
combined_index.add(combined_embeddings)
logging.info(f"FAISS HNSW index for combined text created with {combined_index.ntotal} entries.")

# Save FAISS index to temp
combined_index_path = "/tmp/combined_transformers.faiss"
faiss.write_index(combined_index, combined_index_path)

# Prepare BM25 for sparse retrieval
logging.info("Preparing BM25 for sparse retrieval...")
tokenized_corpus = [text.split() for text in knowledge_base['combined_text']]
bm25 = BM25Okapi(tokenized_corpus)
with open("/tmp/bm25_model.pkl", "wb") as f:
    pickle.dump(bm25, f)
logging.info("BM25 model saved.")

# Store metadata for filtering
logging.info("Storing metadata for filtering...")
metadata_columns = [
    'project', 'fleet', 'subsystem', 'database', 'observation_category', 'problem_code',
    'problem_cause', 'solution_category', 'language', 'failure_class', 'date'
]

metadata = knowledge_base[metadata_columns]
metadata.to_csv("/tmp/metadata.csv", index=False)
logging.info("Metadata saved to '/tmp/metadata.csv'.")

# Upload to Dataiku storage
logging.info("Uploading to Dataiku storage...")
stored_indices = dataiku.Folder("IhvkWeYu")
stored_indices.upload_stream("obs_transformers.faiss", open(obs_index_path, "rb"))
stored_indices.upload_stream("problem_cause_transformers.faiss", open(problem_cause_index_path, "rb"))
stored_indices.upload_stream("problem_code_transformers.faiss", open(problem_code_index_path, "rb"))
stored_indices.upload_stream("solution_transformers.faiss", open(solution_index_path, "rb"))
stored_indices.upload_stream("combined_transformers.faiss", open(combined_index_path, "rb"))
stored_indices.upload_stream("bm25_model.pkl", open("/tmp/bm25_model.pkl", "rb"))
stored_indices.upload_stream("metadata.csv", open("/tmp/metadata.csv", "rb"))
stored_indices.upload_stream("knowledge_base_with_index.csv", open("/tmp/knowledge_base_with_index.csv", "rb"))
logging.info("✅ Hybrid FAISS + BM25 indexing completed successfully!")
stored_indices_info = stored_indices.get_info()
