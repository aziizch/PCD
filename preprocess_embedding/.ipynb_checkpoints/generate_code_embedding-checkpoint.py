"""
Generate embedding for code (High Safety Mode)
Applies 3 constraints:
1. Low Batch Size
2. Limit Total Nodes per file
3. Truncate Content heavily
"""

import os
import sys
import json
import pickle
import tqdm
import torch
import numpy as np
import gc
from transformers import AutoTokenizer, AutoModel

# ===================== SAFETY CRITERIA SETTINGS =====================

# CRITERE 1 : Reduce Batch Size
# Réduit à 2 pour garantir que ça passe même sur des petits GPU
BATCH_SIZE = 4

# CRITERE 2 : Reduce Number of Generated Nodes
# Si un fichier contient plus de 1000 nœuds, on ignore les suivants.
MAX_NODES_PER_FILE = 5000 

# CRITERE 3 : Reduce File/Content Size
# On coupe le texte à 1000 caractères (au lieu de 2000 ou 4096)
MAX_CHARS = 1000 
MAX_TOKENS = 256 # Réduit également la fenêtre de contexte du modèle

# ===================== GPU SETTINGS =====================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== PATHS AUTOMATION =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # Remonte d'un cran

# Chemins cibles
node_content_path = os.path.join(PROJECT_ROOT, "retriever", "content", "restant")
node_embedding_path = os.path.join(PROJECT_ROOT, "retriever", "content", "tmp_node_embedding")

# Détection Modèle
local_model_path = os.path.join(PROJECT_ROOT, "cge-small")
if os.path.exists(local_model_path):
    model_name_or_path = local_model_path
else:
    model_name_or_path = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\cge-small"

# ===================== FUNCTIONS =====================

def truncate_text(text, max_chars=MAX_CHARS):
    """Coupe le texte brutalement pour respecter le Critère 3"""
    if isinstance(text, list):
        return [t[:max_chars] if t else " " for t in text]
    return text[:max_chars] if text else " "

def cleanup():
    """Nettoyage mémoire agressif"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def batch_encode(model_func, texts, tokenizer=None, batch_size=BATCH_SIZE, desc="Encoding"):
    """Encode par très petits paquets"""
    all_embeddings = []
    
    # Pas de barre de progression interne si c'est très court
    iterator = range(0, len(texts), batch_size)
    
    for i in iterator:
        batch_texts = texts[i : i + batch_size]
        batch_texts = truncate_text(batch_texts) # Application Critère 3
        
        try:
            if tokenizer:
                emb = model_func(tokenizer, batch_texts)
            else:
                emb = model_func(None, batch_texts)
                
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            elif isinstance(emb, list):
                emb = np.array(emb)
            
            all_embeddings.append(emb)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n[WARNING] OOM detected. Clearing cache...")
                cleanup()
                # En cas de crash, on skip ce batch spécifique
                continue 
            else:
                raise e
        
        del emb
        cleanup() # Nettoyage à chaque tour vu le Batch Size de 2

    if len(all_embeddings) > 0:
        return np.concatenate(all_embeddings, axis=0)
    return np.array([])

# ===================== LOAD MODEL =====================
tokenizer = None
model = None
encode_func = None

print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] Batch Size: {BATCH_SIZE}")
print(f"[INFO] Max Nodes: {MAX_NODES_PER_FILE}")
print(f"[INFO] Max Chars: {MAX_CHARS}")

try:
    print(f"[INFO] Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
    model.to(DEVICE)
    model.eval()

    def cge_encode_wrapper(tokenizer_arg, text_list):
        with torch.no_grad():
            return model.encode(tokenizer_arg, text_list)

    encode_func = cge_encode_wrapper
    print(f"[SUCCESS] CodeFuse-CGE loaded")

except Exception as e:
    print(f"[WARNING] Fallback to MiniLM. Reason: {e}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    model.eval()

    def minilm_encode_wrapper(_unused_tokenizer, text_list):
        with torch.no_grad():
            return model.encode(text_list, convert_to_numpy=True, batch_size=batch_size)
    encode_func = minilm_encode_wrapper
    tokenizer = None

# ===================== MAIN =====================
if __name__ == "__main__":

    if not os.path.exists(node_content_path):
        print(f"[ERROR] Dossier introuvable : {node_content_path}")
        exit(1)

    os.makedirs(node_embedding_path, exist_ok=True)

    processed = {f.split(".")[0] for f in os.listdir(node_embedding_path) if f.endswith(".pkl")}
    candidate_graphs = os.listdir(node_content_path)
    candidate_graphs.sort()

    pbar = tqdm.tqdm(candidate_graphs, desc="Files", ascii=True, unit="file")

    for filename in pbar:
        instance_id = filename.split(".")[0]
        if instance_id in processed:
            continue
        
        cleanup()

        try:
            with open(os.path.join(node_content_path, filename), "r", encoding="utf-8") as f:
                node_content_dict = json.load(f)
        except Exception:
            continue

        full_node_list = list(node_content_dict["code"].keys())
        if not full_node_list:
            continue

        # --- APPLICATION CRITERE 2 : LIMITATION DES NOEUDS ---
        if len(full_node_list) > MAX_NODES_PER_FILE:
            # On ne garde que les N premiers nœuds
            node_list = full_node_list[:MAX_NODES_PER_FILE]
        else:
            node_list = full_node_list

        # --- PREPARE DATA ---
        code_texts = []
        doc_texts = []
        nodes_with_doc = []
        node_index_map = [] 

        for node in node_list:
            # Récupération et Truncation immédiate
            c_raw = node_content_dict["code"].get(node, " ")
            code_texts.append(truncate_text(c_raw)) # Critère 3 appliqué ici aussi
            node_index_map.append(node)

            if node in node_content_dict.get("doc", {}):
                d_raw = node_content_dict["doc"].get(node, " ")
                doc_texts.append(truncate_text(d_raw))
                nodes_with_doc.append(node)

        # --- RUN INFERENCE ---
        try:
            code_embeddings = batch_encode(encode_func, code_texts, tokenizer, BATCH_SIZE)
            
            doc_embeddings = None
            if doc_texts:
                doc_embeddings = batch_encode(encode_func, doc_texts, tokenizer, BATCH_SIZE)

            # --- SAVE ---
            # Si le batch encoding a échoué partiellement (retourne tableau vide), on évite l'erreur d'index
            if len(code_embeddings) != len(node_index_map):
                # Cas rare où OOM a sauté des batches
                print(f"[WARN] Mismatch length for {filename}. Skipping save.")
                continue

            node_code_embedding_dict = {
                node: code_embeddings[i] for i, node in enumerate(node_index_map)
            }
            
            node_doc_embedding_dict = {}
            if doc_embeddings is not None and len(doc_embeddings) == len(nodes_with_doc):
                 node_doc_embedding_dict = {
                    node: doc_embeddings[i] for i, node in enumerate(nodes_with_doc)
                }

            node_embedding_dict = {
                "code": node_code_embedding_dict,
                "doc": node_doc_embedding_dict
            }

            with open(os.path.join(node_embedding_path, f"{instance_id}.pkl"), "wb") as f:
                pickle.dump(node_embedding_dict, f)

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            cleanup()

    print("\n[SUCCESS] Termine (Mode Safe).")