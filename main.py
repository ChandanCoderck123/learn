import openai
import faiss
import numpy as np
import pandas as pd
import re
import json
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set the embedding model
EMBEDDING_MODEL = "text-embedding-ada-002"

def get_embedding(text):
    text = text.replace("\n", " ").strip()
    try:
        response = openai.Embedding.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        return np.array(response['data'][0]['embedding'], dtype=np.float32)
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def clean_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()

# Load CSV file
csv_path = "SKU_list_of_23-24.csv"

try:
    catalog_df = pd.read_csv(csv_path)
    catalog_df = catalog_df[['SKU', 'Brand', 'Description', 'Qty 2023-24']].fillna("")
    
    # Convert "Qty 2023-24" to integer format
    catalog_df['Qty 2023-24'] = catalog_df['Qty 2023-24'].astype(str).str.replace(',', '').astype(float).fillna(1).astype(int)
    
    product_texts = catalog_df.apply(
        lambda x: f"{x['Brand']} {x['Description']}", axis=1
    )
    product_texts = product_texts.apply(clean_text)

    embeddings_list = []
    valid_indices = []

    for idx, text in enumerate(product_texts):
        emb = get_embedding(text)
        if emb is not None:
            embeddings_list.append(emb)
            valid_indices.append(idx)

    if not embeddings_list:
        raise ValueError("No embeddings generated. Check your API key or data format.")

    embeddings_array = np.vstack(embeddings_list)
    faiss_index = faiss.IndexFlatL2(embeddings_array.shape[1])
    faiss_index.add(embeddings_array)

    catalog_map = {
        i: catalog_df.iloc[valid_indices[i]].to_dict() for i in range(len(valid_indices))
    }
except Exception as e:
    print(f"Error loading CSV or creating embeddings: {e}")
    exit(1)

app = Flask(__name__)

@app.route('/rfq', methods=['POST'])
def rfq_search():
    data = request.get_json()
    if not data or 'rfq' not in data:
        return jsonify({"error": "Invalid request. Provide 'rfq' field in JSON."}), 400

    rfq_input = data['rfq']
    
    # Normalize the input by replacing commas (with or without spaces) with newlines
    rfq_input = re.sub(r'\s*,\s*', '\n', rfq_input.strip())
    
    # Now split by newline
    rfq_lines = rfq_input.split("\n")
    
    matched_products = []

    for line in rfq_lines:
        query = clean_text(line)
        query_embedding = get_embedding(query)
        if query_embedding is None:
            continue
        _, indices = faiss_index.search(query_embedding.reshape(1, -1), 5)
        top_matches = []
        best_match = None

        for rank, match_idx in enumerate(indices[0]):
            if match_idx < 0:
                continue
            matched_row = catalog_map[match_idx]
            
            # Get quantity from "Qty 2023-24" column
            quantity = matched_row.get("Qty 2023-24", 1)
            
            match_entry = {
                "rank": rank + 1,
                "product_id": matched_row["SKU"],
                "brand": matched_row["Brand"],
                "product_name": matched_row["Description"],
                "quantity": quantity
            }
            if rank == 0:
                best_match = match_entry
            top_matches.append(match_entry)

        matched_products.append({
            "original_string": line,
            "best_match": best_match,
            "top_5_matches": top_matches
        })

    return jsonify(matched_products), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
