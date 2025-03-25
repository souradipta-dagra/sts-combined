import dataiku
from dataiku import pandasutils as pdu
import pandas as pd
import faiss
import numpy as np
import pickle
from rank_bm25 import BM25Okapi
import os
import tempfile
from sentence_transformers import SentenceTransformer

# Get Dataiku folder handle
handle = dataiku.Folder('g6KmFVt0')

# Load FAISS index using a temporary file
with handle.get_download_stream('knowledge_base_index.faiss') as f:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(f.read())  # Write bytes to a temp file
        tmp_file_path = tmp_file.name  # Get temp file path

index = faiss.read_index(tmp_file_path)  # Read FAISS index from temp file
os.remove(tmp_file_path)  # Cleanup temp file
print(f"Loaded FAISS index with {index.ntotal} entries.")

# Load BM25 model
with handle.get_download_stream('bm25_model.pkl') as f:
    bm25 = pickle.load(f)
print("Loaded BM25 model.")

# Load metadata
with handle.get_download_stream('metadata.csv') as f:
    metadata = pd.read_csv(f, low_memory=False)
print(f"Loaded metadata with {len(metadata)} rows.")

# Load knowledge base with index for text reference
with handle.get_download_stream('knowledge_base_with_index.csv') as f:
    knowledge_base = pd.read_csv(f, low_memory=False)
print(f"Loaded knowledge base with {len(knowledge_base)} entries.")

# Ensure columns category_id, obs_id, sol_category_id exist and are filled as blank
for col in ['category_id', 'obs_id', 'sol_category_id']:
    if col not in knowledge_base.columns:
        knowledge_base[col] = ""

# Load multi-vector embeddings
with handle.get_download_stream('multi_embeddings.pkl') as f:
    multi_embeddings = pd.read_pickle(f)
print("Loaded multi-vector embeddings.")

# Load pre-trained Sentence Transformer model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Function to perform dense retrieval using FAISS
def dense_retrieval(query_embedding, top_k=100):
    D, I = index.search(np.array([query_embedding]), top_k)
    return I[0], D[0]

# Function to perform sparse retrieval using BM25
def sparse_retrieval(query, top_k=100):
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_k_indices = np.argsort(bm25_scores)[::-1][:top_k]
    return top_k_indices, bm25_scores[top_k_indices]

# Function to perform multi-vector retrieval
def multi_vector_retrieval(query_observation_embedding, top_k=100):
    observation_embeddings = np.vstack(multi_embeddings['observation_embedding'])

    # Normalize embeddings for better FAISS search
    faiss.normalize_L2(observation_embeddings)

    # Create FAISS index for observation embeddings
    dimension = query_observation_embedding.shape[0]
    observation_index = faiss.IndexFlatL2(dimension)
    observation_index.add(observation_embeddings)

    # Perform dense retrieval for observation embedding
    D_obs, I_obs = observation_index.search(np.array([query_observation_embedding]), top_k)

    combined_indices = list(I_obs[0])
    combined_scores = list(D_obs[0])

    return combined_indices, combined_scores

# Function to combine and rerank results from dense, sparse, and multi-vector retrieval
def hybrid_search(query, top_k=100):
    # Generate embeddings for the query observation
    query_embedding = embedding_model.encode(query)
    query_observation_embedding = embedding_model.encode(query)

    # Perform dense retrieval
    dense_indices, dense_scores = dense_retrieval(query_embedding, top_k)

    # Perform sparse retrieval
    sparse_indices, sparse_scores = sparse_retrieval(query, top_k)

    # Perform multi-vector retrieval
    multi_indices, multi_scores = multi_vector_retrieval(query_observation_embedding, top_k)

    # Combine results from dense, sparse, and multi-vector retrieval
    combined_indices = list(dense_indices) + list(sparse_indices) + list(multi_indices)
    combined_scores = list(dense_scores) + list(sparse_scores) + list(multi_scores)

    # Rerank combined results
    combined_results = sorted(zip(combined_indices, combined_scores), key=lambda x: x[1], reverse=True)
    combined_results = combined_results[:top_k]

    return combined_results

# Function to filter results based on metadata
def filter_results(results, metadata_filters):
    filtered_results = []
    for index, score in results:
        row = knowledge_base.iloc[index]
        match = all(row[key] == value for key, value in metadata_filters.items())
        if match:
            filtered_results.append((row, score))
    return filtered_results

def handle_query(query, metadata_filters, top_k=100):
    # Perform hybrid search
    combined_results = hybrid_search(query, top_k)

    # Filter results based on metadata
    filtered_results = filter_results(combined_results, metadata_filters)

    # Format results to include observation and solution
    final_results = {
        "response": {
            "observation_results": []
        }
    }
    for rank, (result, score) in enumerate(filtered_results):
        formatted_result = {
            "category_id": result.get('category_id', ""),
            "project": result.get('project', ""),
            "country": result.get('country', ""),
            "fleet": result.get('fleet', ""),
            "subsystem": result.get('subsystem', ""),
            "database": result.get('database', ""),
            "observation_category": result.get('observation_category', ""),
            "obs_id": result.get('obs_id', ""),
            "observation": result.get('observation', ""),
            "failure_class": result.get('failure_class', ""),
            "problem_code": result.get('problem_code', ""),
            "problem_cause": result.get('problem_cause', ""),
            "problem_remedy": result.get('problem_remedy', ""),
            "functional_location": result.get('functional_location', ""),
            "notifications_number": result.get('notifications_number', ""),
            "date": result.get('date', ""),
            "solution_category": result.get('solution_category', ""),
            "solution": result.get('solution', ""),
            "pbs_code": result.get('pbs_code', ""),
            "symptom_code": result.get('symptom_code', ""),
            "root_cause": result.get('root_cause', ""),
            "document_link": result.get('document_link', ""),
            "language": result.get('language', ""),
            "resource": result.get('resource', 0),
            "min_resources_need": result.get('min_resources_need', ""),
            "max_resource_need": result.get('max_resource_need', ""),
            "the_most_frequent_value_for_resource": result.get('the_most_frequent_value_for_resource', ""),
            "time": result.get('time', ""),
            "min_time_per_one_person": result.get('min_time_per_one_person', ""),
            "max_time_per_one_person": result.get('max_time_per_one_person', ""),
            "average_time": result.get('average_time', ""),
            "frequency_obs": result.get('frequency_obs', 0.0),
            "frequency_sol": result.get('frequency_sol', 0.0),
            "min_resources_need_sol": result.get('min_resources_need_sol', ""),
            "max_resource_need_sol": result.get('max_resource_need_sol', ""),
            "the_most_frequent_value_for_resource_sol": result.get('the_most_frequent_value_for_resource_sol', ""),
            "min_time_per_one_person_sol": result.get('min_time_per_one_person_sol', ""),
            "max_time_per_one_person_sol": result.get('max_time_per_one_person_sol', ""),
            "average_time_sol": result.get('average_time_sol', ""),
            "sol_category_id": result.get('sol_category_id', ""),
            "ranking": rank + 1
        }
        final_results["response"]["observation_results"].append(formatted_result)
    
    return final_results

if __name__ == "__main__":
    test_query = "Preventive Maintenullce Washing Machine"
    metadata_filters = {
#      "project": "VLINE RRSMC"
    }

    results = handle_query(test_query, metadata_filters, top_k=150)
    print(results)
