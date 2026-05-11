"""
serialize_subgraph.py
(Version SAFE: Anti-Freeze & Debug Logs)
"""

import os
import json
import tqdm
import networkx as nx
import pandas as pd
import time

# --- IMPORTS CODEFUSE ---
try:
    from codegraph_parser.python.codegraph_python_local import parse, NodeType, EdgeType
    from utils import codegraph_to_nxgraph
except ImportError:
    print("❌ ERROR: Modules CodeFuse introuvables.")
    exit(1)


################################# CONFIG #################################
BASIC_INFO_PATH = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\preprocess_embedding\test_lite_basic_info.json"
GRAPH_DIR = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\version initial codefuse\dataset\swe-bench-lite"
SUBGRAPH_DICT_PATH = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\retriever\subgraph_nodes2.json"
SAVE_DIR = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\retriever\subgraphs_final2"

FILE_LEVEL_EXPAND = True
################################# CONFIG #################################


############################# utils #############################
def safe_get_node_by_id(graph, node_id):
    if node_id is None: return None
    try:
        nid = int(node_id)
    except (TypeError, ValueError):
        nid = node_id
    return graph.get_node_by_id(nid)

def get_contained_node(graph_nx: nx.MultiDiGraph, node):
    c_node_list = []
    try:
        successors = list(graph_nx.successors(node))
    except Exception:
        return []

    for suc_node in successors:
        try:
            edge_data = graph_nx[node][suc_node]
            edge_attr = edge_data[0] if 0 in edge_data else list(edge_data.values())[0]
            if edge_attr.get("type") == EdgeType.CONTAINS:
                c_node_list.append(suc_node)
        except Exception:
            continue
    return c_node_list

# --- CORRECTION MAJEURE ICI (ANTI BOUCLE INFINIE) ---
def get_inner_nodes_safe(graph_nx: nx.MultiDiGraph, start_node):
    """
    Récupère tous les descendants via CONTAINS de manière itérative et sécurisée.
    Utilise un 'visited' set pour éviter les boucles infinies.
    """
    visited = set([start_node])
    stack = [start_node]
    inner_nodes_all = []

    while stack:
        current_node = stack.pop()
        
        children = get_contained_node(graph_nx, current_node)
        
        for child in children:
            if child not in visited:
                visited.add(child)
                stack.append(child)
                inner_nodes_all.append(child)
    
    return inner_nodes_all


def serialize_subgraph(graph_nx: nx.MultiDiGraph, file_stem: str) -> bool:
    node_list = []
    for node in graph_nx.nodes():
        try:
            node_data = node.to_dict()
        except AttributeError:
            node_data = {"id": str(node), "type": "unknown"}
        node_list.append(node_data)

    edge_list = []
    for u, v in graph_nx.edges():
        try:
            edge_data = graph_nx[u][v]
            edge_attr = edge_data[0] if 0 in edge_data else list(edge_data.values())[0]
            edge_type = edge_attr.get("type")
            edge_type_name = edge_type.name.lower() if edge_type else "unknown"

            edge_list.append({
                "edgeType": edge_type_name,
                "source": u.node_id if hasattr(u, 'node_id') else str(u),
                "target": v.node_id if hasattr(v, 'node_id') else str(v),
            })
        except Exception:
            continue

    graph_json = {"nodes": node_list, "edges": edge_list}

    os.makedirs(os.path.dirname(file_stem), exist_ok=True)
    with open(file_stem + ".json", "w", encoding="utf-8") as f:
        json.dump(graph_json, f, ensure_ascii=False, indent=2)
    return True

def get_real_filename(item):
    repo = item["repo"]
    base_commit = (item.get("base_commit") or item.get("base_sha") or item.get("sha") or item.get("commit"))
    if base_commit is None: return None
    repo = str(repo).replace("/", "#", 1)
    base_commit = str(base_commit)
    return f"{repo}#{base_commit}.graph.json"

############################# Main #############################
if __name__ == "__main__":
    print("🚀 Démarrage Safe-Mode...")
    
    # Mapping ID -> Filename
    df_info = pd.read_json(BASIC_INFO_PATH, orient='index') if os.path.exists(BASIC_INFO_PATH) else pd.DataFrame()
    id_to_filename = {}
    if "instance_id" in df_info.columns:
        for _, row in df_info.iterrows():
            fname = get_real_filename(row)
            if fname: id_to_filename[str(row["instance_id"])] = fname

    with open(SUBGRAPH_DICT_PATH, "r", encoding="utf-8") as f:
        subgraph_nodes_dict = json.load(f)

    print(f"📂 {len(subgraph_nodes_dict)} sous-graphes à traiter.")
    os.makedirs(SAVE_DIR, exist_ok=True)

    success_count = 0
    pbar = tqdm.tqdm(subgraph_nodes_dict.items(), total=len(subgraph_nodes_dict), ascii=True)

    for instance_id, node_ids in pbar:
        instance_id = str(instance_id)
        
        # LOGS DE DEBUG
        # pbar.set_description(f"Processing {instance_id}")

        out_stem = os.path.join(SAVE_DIR, instance_id)
        if os.path.exists(out_stem + ".json"):
            continue

        real_filename = id_to_filename.get(instance_id)
        graph_path = os.path.join(GRAPH_DIR, real_filename) if real_filename else ""
        
        if not graph_path or not os.path.exists(graph_path):
            # Fallback
            candidates = [os.path.join(GRAPH_DIR, f"{instance_id}.json"), os.path.join(GRAPH_DIR, f"{instance_id}.graph.json")]
            for c in candidates:
                if os.path.exists(c):
                    graph_path = c
                    break
        
        if not os.path.exists(graph_path):
            continue

        try:
            # 1. Parsing
            graph = parse(graph_path)
            graph_nx = codegraph_to_nxgraph(graph)

            all_ids = list(node_ids) if isinstance(node_ids, list) else []

            # 2. Expanding (C'est souvent ici que ça bloquait)
            if FILE_LEVEL_EXPAND:
                initial_len = len(all_ids)
                new_ids = []
                for nid in list(all_ids):
                    n = safe_get_node_by_id(graph, nid)
                    if n is None: continue
                    if hasattr(n, 'get_type') and n.get_type() == NodeType.FILE:
                        # Utilisation de la fonction SAFE
                        inner_nodes = get_inner_nodes_safe(graph_nx, n)
                        for inner in inner_nodes:
                            new_ids.append(inner.node_id)
                all_ids.extend(new_ids)

            # 3. Resolving Objects
            node_objs = []
            seen = set()
            for nid in all_ids:
                key = str(nid)
                if key in seen: continue
                seen.add(key)
                n = safe_get_node_by_id(graph, nid)
                if n: node_objs.append(n)

            if not node_objs:
                serialize_subgraph(nx.MultiDiGraph(), out_stem)
                continue

            # 4. Saving
            subgraph = graph_nx.subgraph(node_objs)
            serialize_subgraph(subgraph, out_stem)
            success_count += 1
            
        except Exception as e:
            pbar.write(f"❌ Error on {instance_id}: {e}")
            continue

    print(f"\n✅ Terminé ! {success_count} fichiers générés.")