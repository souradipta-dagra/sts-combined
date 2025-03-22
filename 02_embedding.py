# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle
import pandas as pd
from tqdm import tqdm

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Load the dataset
knowledge_base = dataiku.Dataset("sts_combined_data").get_dataframe()

# Identify problematic columns
mixed_type_cols = [col for col in knowledge_base.columns if knowledge_base[col].map(type).nunique() > 1]

# Convert only these columns to strings
for col in mixed_type_cols:
    knowledge_base[col] = knowledge_base[col].astype(str)

# Add combined_text column to knowledge_base DataFrame
knowledge_base['text'] = ["observation: " + obs + " solution: " + sol
                                  for obs, sol in zip(knowledge_base['observation'],
                                                     knowledge_base['solution'])]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Ensure required columns are present
required_columns = ['text', 'project', 'language', 'observation', 'solution']
for col in required_columns:
    if col not in knowledge_base.columns:
        raise ValueError(f"Missing required column: {col}")

# Load Sentence Transformer model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 1. Generate single-vector embeddings for the 'text' column
print("Generating single-vector embeddings for 'text'...")
single_embeddings = embedding_model.encode(
    knowledge_base['text'].tolist(), show_progress_bar=True, batch_size=32
)
single_embeddings = np.array(single_embeddings).astype('float32')  # FAISS expects float32

# Normalize embeddings for better FAISS search
faiss.normalize_L2(single_embeddings)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 2. Build a FAISS HNSW index for dense retrieval
dimension = single_embeddings.shape[1]
index = faiss.IndexHNSWFlat(dimension, 32)
index.hnsw.efConstruction = 40 
index.add(single_embeddings)
print(f"FAISS HNSW index created with {index.ntotal} entries.")

# Save FAISS index to temp
faiss.write_index(index, "/tmp/knowledge_base_index.faiss")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 3. Prepare BM25 for sparse retrieval
print("Preparing BM25 for sparse retrieval...")
tokenized_corpus = [text.split() for text in knowledge_base['text']]
bm25 = BM25Okapi(tokenized_corpus)
with open("/tmp/bm25_model.pkl", "wb") as f:
    pickle.dump(bm25, f)
print("BM25 model saved.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 4. Store metadata for filtering
metadata_columns = [
    'project', 'fleet', 'subsystem', 'database', 'observationcategory', 'problemcode',
    'problemcause', 'solutioncategory', 'language', 'failureclass', 'date', 'observation', 'solution'
]
metadata = knowledge_base[metadata_columns]
metadata.to_csv("/tmp/metadata.csv", index=False)
print("Metadata saved to '/tmp/metadata.csv'.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 5. Generate and store separate embeddings for multivector retrieval
print("Generating separate embeddings for observation and solution...")
multi_embeddings = {
    'observation': embedding_model.encode(
        knowledge_base['observation'].tolist(), show_progress_bar=True, batch_size=32
    ),
    'solution': embedding_model.encode(
        knowledge_base['solution'].tolist(), show_progress_bar=True, batch_size=32
    )
}

# Convert to DataFrame and save
multi_df = pd.DataFrame({
    'index': knowledge_base.index,
    'observation_embedding': list(multi_embeddings['observation']),
    'solution_embedding': list(multi_embeddings['solution'])
})

multi_df.to_pickle("/tmp/multi_embeddings.pkl")
print("Separate embeddings saved to '/tmp/multi_embeddings.pkl'.")

# Save the original knowledge base with index for reference
knowledge_base.to_csv("/tmp/knowledge_base_with_index.csv", index=True)
print("Knowledge base with index saved.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Upload to Dataiku storage
stored_indices = dataiku.Folder("JgFDry5D")
stored_indices.upload_stream("knowledge_base_index.faiss", open("/tmp/knowledge_base_index.faiss", "rb"))
stored_indices.upload_stream("bm25_model.pkl", open("/tmp/bm25_model.pkl", "rb"))
stored_indices.upload_stream("metadata.csv", open("/tmp/metadata.csv", "rb"))
stored_indices.upload_stream("multi_embeddings.pkl", open("/tmp/multi_embeddings.pkl", "rb"))
stored_indices.upload_stream("knowledge_base_with_index.csv", open("/tmp/knowledge_base_with_index.csv", "rb"))

print("✅ Hybrid FAISS + BM25 indexing completed successfully!")
stored_indices_info = stored_indices.get_info()