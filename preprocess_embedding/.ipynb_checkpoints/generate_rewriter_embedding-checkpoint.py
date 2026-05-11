"""
generate embedding for Queries from Rewriter's Inferer
(Windows Encoding Fixed)
"""

from transformers import AutoTokenizer, AutoModel
import torch
import os
import numpy as np
import pandas as pd
import tqdm
import json
import pickle

# custom
import argparse, logging

# ===================== PATHS =====================
rewriter_output_path = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\rewriter\rewriter_output.json"
rewriter_embedding_path = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\rewriter\rewriter_embedding.pkl"

# ===================== LOAD MODEL =====================
model_name_or_path = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\cge-small"

print(f"[INFO] Loading model from {model_name_or_path}...")
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, truncation_side='right', padding_side='right')

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
model.to('cpu')
print(f"[INFO] Model loaded on {device}")

if __name__ == "__main__":

    print(f"[INFO] Reading JSON file...")
    
    # --- CORRECTION ICI : Ajout de encoding='utf-8' ---
    try:
        with open(rewriter_output_path, 'r', encoding='utf-8') as file:
            rewriter_output_dict = json.load(file)
    except FileNotFoundError:
        print(f"[ERROR] Le fichier n'existe pas : {rewriter_output_path}")
        exit(1)

    print(f"[INFO] Found {len(rewriter_output_dict)} queries to process.")

    query_embedding_dict = {}

    for instance_id in tqdm.tqdm(rewriter_output_dict, desc="Encoding Queries", ascii=True):
        
        # Sécurité si la clé 'query' manque ou est vide
        query = rewriter_output_dict[instance_id].get("query", "")

        if not query or len(query) == 0:
            continue

        # Encodage (Pas besoin de gros batching ici car les queries sont courtes)
        with torch.no_grad():
            emb = model.encode(tokenizer, query)
            
            # Conversion en Numpy pour sauvegarder (plus léger)
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            
            query_embedding_dict[instance_id] = emb

    print(f"[INFO] Saving embeddings to {rewriter_embedding_path}...")
    with open(rewriter_embedding_path, 'wb') as f:
        pickle.dump(query_embedding_dict, f)

    print("[SUCCESS] Done.")

