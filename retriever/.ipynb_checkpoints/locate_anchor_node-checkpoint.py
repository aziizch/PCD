"""
基于 rapidfuzz + faiss 进行 anchor node 定位
(Version corrigée avec Logs de débogage et Sauvegarde forcée)
"""

import os
# Création des dossiers manquants AVANT les imports qui pourraient planter
output_save_path = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\retriever\anchor_node2.json"
os.makedirs(os.path.dirname(output_save_path), exist_ok=True)

from rapidfuzz import process, fuzz
import pandas as pd
import json
from tqdm import tqdm
import pickle
import numpy as np
import faiss

# Assurez-vous que ces imports fonctionnent dans votre environnement
try:
    from codegraph_parser.python.codegraph_python_local import parse, NodeType
    from utils import codegraph_to_nxgraph
except ImportError:
    print("⚠️ [ATTENTION] Modules 'codegraph_parser' ou 'utils' introuvables.")
    print("Assurez-vous d'être à la racine du projet ou d'avoir configuré le PYTHONPATH.")

def extract_info(item):
    return item[1]


################################# Extractor #################################
def get_extractor_anchor(graph, entity_query, keywords_query):
    all_nodes = graph.get_nodes()

    cand_name_list = []
    cand_path_name_list = []

    for node in all_nodes:
        node_type = node.get_type()
        if node_type in [NodeType.REPO, NodeType.PACKAGE]:
            continue

        if not hasattr(node, "name"):
            continue

        cand_name_list.append((node.node_id, node.name))

        if node_type == NodeType.FILE:
            if node.path:
                name_with_path = node.path + "/" + node.name
            else:
                name_with_path = node.name
            cand_path_name_list.append((node.node_id, name_with_path))

    cand_name_all = []
    cand_path_name_all = []

    # Combine queries
    full_queries = entity_query + keywords_query
    
    # Sécurité si les listes sont vides
    if not full_queries:
        return set()

    for query in full_queries:
        if not query: continue

        if "/" in query:
            cand_path_name = process.extract(
                (-1, query),
                cand_path_name_list,
                scorer=fuzz.WRatio,
                limit=3,
                processor=extract_info
            )
            cand_path_name_all.append(cand_path_name)

        query_wo_path = query.split("/")[-1]
        cand_name = process.extract(
            (-1, query_wo_path),
            cand_name_list,
            scorer=fuzz.WRatio,
            limit=3,
            processor=extract_info
        )
        cand_name_all.append(cand_name)

    res = set()
    for query in cand_name_all:
        for item in query:
            res.add(item[0][0])

    for query in cand_path_name_all:
        for item in query:
            res.add(item[0][0])

    return res


################################# Inferer #################################
def get_inferer_anchor(query_emb, node_embedding, k=15):

    node2id = {}
    id2node = {}
    cand_vec = []

    raw_node_embedding = node_embedding["code"]
    
    if not raw_node_embedding:
        return []

    for i, node_id in enumerate(raw_node_embedding):
        node2id[node_id] = i
        id2node[i] = node_id
        cand_vec.append(raw_node_embedding[node_id])

    # Node embeddings → numpy float32
    cand_vec_np = np.array(cand_vec, dtype="float32")

    # Query embedding → numpy float32 + 2D
    query_emb = np.array(query_emb, dtype="float32")
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)

    # Vérification dimensions
    if query_emb.shape[1] != cand_vec_np.shape[1]:
        print(f"⚠️ Dim mismatch: Query {query_emb.shape} vs Nodes {cand_vec_np.shape}. Skipping FAISS.")
        return []

    d = cand_vec_np.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(cand_vec_np)

    k = min(k, cand_vec_np.shape[0])
    if k == 0: return []
    
    D, I = index.search(query_emb, k)

    anchor_nodes = []
    for row in I:
        anchor_nodes.append([id2node[i] for i in row])

    # Aplatir la liste de listes si nécessaire ou retourner la première liste
    # Ici on retourne une liste simple des IDs trouvés (Row 0 car 1 query)
    return anchor_nodes[0] if anchor_nodes else []


################################# Utils #################################
def get_graph_file_name(item):
    # Assurez-vous que l'extension est correcte (.json)
    fname = item["instance_id"]
    if not fname.endswith(".json"):
        fname += ".json"
    return fname


################################# Main #################################
if __name__ == "__main__":

    print("🚀 Démarrage du script Anchor Node...")

    # ---------- PATHS ----------
    # Input 1: Le fichier de test info
    test_basic_path = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\preprocess_embedding\test_lite_basic_info.json"
    
    # Input 2: Le dossier contenant les JSONs des graphes (Générés par le parser)
    # ATTENTION: J'ai changé ce chemin car 'dataset\swe-bench-lite' contient souvent le code brut, pas les graphes json.
    # Remettez l'ancien si vous êtes sûr de vous, mais vérifiez que les .json sont dedans !
    graph_data_path = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\version initial codefuse\dataset\swe-bench-lite"
    
    # Input 3: Les embeddings des noeuds
    node_embedding_path = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\retriever\content\tmp_node_embedding2"
    
    # Input 4: Output du rewriter
    rewriter_output_path = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\rewriter\rewriter_output.json"
    
    # Input 5: Embedding des queries
    query_embedding_path = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\rewriter\rewriter_embedding.pkl"

    # OUTPUT FINAL
    final_output_path = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\retriever\anchor_node2.json"

    # ---------- Verification ----------
    if not os.path.exists(test_basic_path):
        print(f"❌ ERREUR: Fichier introuvable {test_basic_path}")
        exit(1)
    if not os.path.exists(node_embedding_path):
        print(f"❌ ERREUR: Dossier Embeddings introuvable {node_embedding_path}")
        exit(1)
    if not os.path.exists(graph_data_path):
        print(f"❌ ERREUR: Dossier Graphes introuvable {graph_data_path}")
        print("💡 Astuce: Vérifiez si vos graphes .json sont dans 'retriever/content/output'")
        # exit(1) # On ne quitte pas, mais ça va probablement échouer

    # ---------- Load files ----------
    print("📥 Chargement des fichiers de configuration...")
    
    test_basic_df = pd.read_json(test_basic_path, orient="index")
    test_basic_df["base_commit"] = test_basic_df.apply(lambda item: get_graph_file_name(item), axis=1)

    with open(rewriter_output_path, "r", encoding="utf-8") as f:
        rewriter_output = json.load(f)

    with open(query_embedding_path, "rb") as f:
        query_embedding = pickle.load(f)

    print(f"ℹ️ {len(test_basic_df)} instances à traiter.")

    anchor_node_dict = {}

    # ---------- Main loop ----------
    for _, item in tqdm(test_basic_df.iterrows(), total=len(test_basic_df), ascii=True):

        instance_id = item.instance_id
        
        # Check 1: Query Embeddings
        if instance_id not in query_embedding:
            # print(f"⚠️ Skip {instance_id}: Pas d'embedding de query")
            continue
            
        # Check 2: Graph File
        graph_file = os.path.join(graph_data_path, item.graph_file)
        if not os.path.exists(graph_file):
            print(f"⚠️ Skip {instance_id}: Fichier graphe introuvable dans {graph_data_path}")
            continue

        # Check 3: Node Embeddings
        node_emb_file = os.path.join(node_embedding_path, f"{instance_id}.pkl")
        if not os.path.exists(node_emb_file):
            # print(f"⚠️ Skip {instance_id}: Fichier .pkl node embedding introuvable")
            continue

        try:
            # Load query embedding
            query_emb = np.array(query_embedding[instance_id], dtype="float32")
            if query_emb.ndim == 1:
                query_emb = query_emb.reshape(1, -1)

            # Load graph
            graph = parse(graph_file)
            
            # NOTE: Si vous avez besoin de networkx, décommentez:
            # graph_nx = codegraph_to_nxgraph(graph)

            # Rewriter output check
            if instance_id not in rewriter_output:
                continue
                
            entity_query = rewriter_output[instance_id].get("code_entity", [])
            keyword_query = rewriter_output[instance_id].get("keyword", [])

            # Load node embeddings
            with open(node_emb_file, "rb") as f:
                node_embedding = pickle.load(f)

            # --- PROCESS ---
            res_extractor = get_extractor_anchor(graph, entity_query, keyword_query)
            res_inferer = get_inferer_anchor(query_emb, node_embedding)

            anchor_node_dict[instance_id] = {
                "extractor_anchor_nodes": list(res_extractor),
                "inferer_anchor_nodes": res_inferer
            }

            # SAVE PROGRESSIVELY (Évite de tout perdre si crash)
            # On réécrit le fichier à chaque tour (un peu lent mais sûr)
            with open(final_output_path, "w", encoding="utf-8") as f:
                json.dump(anchor_node_dict, f, indent=2)

        except Exception as e:
            print(f"❌ Erreur sur {instance_id}: {e}")
            continue

    print(f"\n✅ Terminé ! Fichier sauvegardé ici :")
    print(final_output_path)
    print(f"Nombre d'instances traitées avec succès : {len(anchor_node_dict)}")