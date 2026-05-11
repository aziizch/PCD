"""
SubGraph Generation
(Version sécurisée contre RecursionError)
"""
import json
import os
import tqdm
import pandas as pd
import sys

# Augmenter la limite de récursion (sécurité)
sys.setrecursionlimit(2000)

from codegraph_parser.python.codegraph_python_local import parse, NodeType, EdgeType
from utils import codegraph_to_nxgraph


################################# Robust basic-info loader #################################
def load_basic_info_df(path: str) -> pd.DataFrame:
    text = open(path, "r", encoding="utf-8").read().strip()
    if not text:
        raise ValueError(f"Empty file: {path}")

    dec = json.JSONDecoder()
    objs = []
    i = 0
    n = len(text)

    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        obj, end = dec.raw_decode(text, i)
        objs.append(obj)
        i = end

    def to_records(o):
        if isinstance(o, list): return o
        if isinstance(o, dict) and all(isinstance(v, dict) for v in o.values()): return list(o.values())
        if isinstance(o, dict): return [o]
        return [{"value": o}]

    records = []
    for o in objs: records.extend(to_records(o))

    df = pd.DataFrame.from_records(records)
    if df.shape[1] == 1:
        col0 = df.columns[0]
        if df[col0].map(lambda x: isinstance(x, dict)).all():
            df = pd.DataFrame.from_records(df[col0].tolist())

    return df


################################# CORRECTION RECURSION #################################
def get_path_to_repo(node, pre_node_dict, graph_nx, visited=None):
    """
    Get path to repo safely.
    :param visited: Set of nodes currently in the recursion stack (to detect cycles)
    """
    if visited is None:
        visited = set()

    # Base case: Repo node
    if node.get_type() == NodeType.REPO:
        return [node]

    # Memoization
    if node.node_id in pre_node_dict:
        return pre_node_dict[node.node_id]

    # Cycle detection
    if node.node_id in visited:
        # print(f"Cycle detected at {node.node_id}, breaking recursion.")
        return []
    
    visited.add(node.node_id)

    pre_nodes = []
    
    # Try/Except pour la sécurité si le graphe est malformé
    try:
        predecessors = list(graph_nx.predecessors(node))
    except Exception:
        visited.remove(node.node_id)
        return []

    for pre_node in predecessors:
        try:
            edge_data = graph_nx[pre_node][node]
            # NetworkX MultiDiGraph stores edges in a dict keyed by key (usually 0)
            edge_attr = edge_data[0] if 0 in edge_data else list(edge_data.values())[0]

            if edge_attr.get("type") == EdgeType.CONTAINS:
                pre_nodes.append(pre_node)
                
                # Recursive call ONLY if not Repo
                if pre_node.get_type() != NodeType.REPO:
                    # Pass a copy of visited or manage add/remove properly
                    path = get_path_to_repo(pre_node, pre_node_dict, graph_nx, visited)
                    pre_nodes.extend(path)
                break
        except (KeyError, IndexError):
            continue

    visited.remove(node.node_id) # Backtracking
    
    # Only cache if we found something relevant (optional optimization)
    if pre_nodes:
        pre_node_dict[node.node_id] = pre_nodes
        
    return pre_nodes


def reconstruct_graph(subgraph_nodes, graph_nx, pre_node_dict):
    """
    Reconstruct connected CodeGraph based on seed nodes.
    """
    all_nodes = set()
    
    for node in subgraph_nodes:
        if node is None:
            continue
        
        all_nodes.add(node)
        
        # New safe call
        path = get_path_to_repo(node, pre_node_dict, graph_nx)
        all_nodes.update(path)

    return graph_nx.subgraph(list(all_nodes))


################################# BFS代码 #################################
def bfs_expand_file(graph_nx, subgraph_nodes, hops=1):
    seed_nodes = [n for n in subgraph_nodes if n is not None]
    visited_node = set(n.node_id for n in seed_nodes)
    nhops_neighbors = set(n.node_id for n in seed_nodes)

    for _ in range(hops):
        tmp_seed_nodes = []
        for cur in seed_nodes:
            if cur is None: continue

            # Successors
            try:
                for nxt in graph_nx.successors(cur):
                    if nxt is None: continue
                    nhops_neighbors.add(nxt.node_id)
                    if nxt.node_id not in visited_node and nxt.get_type() == NodeType.FILE:
                        visited_node.add(nxt.node_id)
                        tmp_seed_nodes.append(nxt)
            except Exception: pass

            # Predecessors
            try:
                for prv in graph_nx.predecessors(cur):
                    if prv is None: continue
                    nhops_neighbors.add(prv.node_id)
                    if prv.node_id not in visited_node and prv.get_type() == NodeType.FILE:
                        visited_node.add(prv.node_id)
                        tmp_seed_nodes.append(prv)
            except Exception: pass

        seed_nodes = tmp_seed_nodes

    return nhops_neighbors


################################# 辅助函数 #################################
def get_graph_file_name(item: pd.Series) -> str:
    repo = item["repo"]
    base_commit = (item.get("base_commit") or item.get("base_sha") or item.get("sha") or item.get("commit"))
    if base_commit is None:
        raise KeyError(f"Missing commit field in row: {item.to_dict()}")

    repo = str(repo).replace("/", "#", 1)
    base_commit = str(base_commit)
    return f"{repo}#{base_commit}.graph.json"


def safe_get_node_by_id(graph, node_id):
    if node_id is None: return None
    try:
        nid = int(node_id)
    except (TypeError, ValueError):
        nid = node_id
    return graph.get_node_by_id(nid)


################################# Main #################################
if __name__ == "__main__":
    
    # --- CHEMINS (VERIFIEZ CES CHEMINS) ---
    basic_info_path = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\preprocess_embedding\test_lite_basic_info.json"
    
    # ATTENTION : Utilisez le dossier contenant les VRAIS graphes (.graph.json)
    graph_data_dir = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\version initial codefuse\dataset\swe-bench-lite"
    
    anchor_node_path = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\retriever\anchor_node2.json"
    output_path = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\retriever\subgraph_nodes2.json"

    # Load basic info
    test_basic_df = load_basic_info_df(basic_info_path)
    test_basic_df["graph_file"] = test_basic_df.apply(get_graph_file_name, axis=1)

    # Load anchor nodes
    with open(anchor_node_path, "r", encoding="utf-8") as f:
        anchor_node_dict = json.load(f)

    subgraph_id_dict = {}

    print(f"🚀 Processing {len(test_basic_df)} instances...")
    
    for _, item in tqdm.tqdm(test_basic_df.iterrows(), total=len(test_basic_df), ascii=True):
        instance_id = str(item.instance_id)

        if instance_id not in anchor_node_dict:
            continue

        graph_file = item.graph_file
        tmp_graph_data_path = os.path.join(graph_data_dir, graph_file)
        
        if not os.path.exists(tmp_graph_data_path):
            # print(f"Skip {instance_id}: Graph file not found")
            continue

        try:
            # Parse graph
            graph = parse(tmp_graph_data_path)
            graph_nx = codegraph_to_nxgraph(graph)

            # Get anchor node ids
            anchor_nodes_raw = anchor_node_dict[instance_id]
            extractor_anchors = anchor_nodes_raw.get("extractor_anchor_nodes", [])
            inferer_anchors = []
            
            # Handle inferer list of lists
            raw_inf = anchor_nodes_raw.get("inferer_anchor_nodes", [])
            if raw_inf and isinstance(raw_inf[0], list):
                 for sublist in raw_inf: inferer_anchors.extend(sublist)
            else:
                inferer_anchors = raw_inf

            anchor_node_ids = list(set(extractor_anchors + inferer_anchors))

            # Convert ids -> nodes
            anchor_nodes = []
            for nid in anchor_node_ids:
                n = safe_get_node_by_id(graph, nid)
                if n is not None:
                    anchor_nodes.append(n)

            if not anchor_nodes:
                subgraph_id_dict[instance_id] = []
                continue

            # BFS expand + reconstruct
            expanded_node_ids = bfs_expand_file(graph_nx, anchor_nodes, hops=2)

            expanded_nodes = []
            for nid in expanded_node_ids:
                n = safe_get_node_by_id(graph, nid)
                if n is not None:
                    expanded_nodes.append(n)

            pre_node_dict = {}
            # LE POINT CRITIQUE EST ICI
            subgraph = reconstruct_graph(expanded_nodes, graph_nx, pre_node_dict)

            # Collect FILE node ids
            file_node_ids = [
                n.node_id for n in subgraph.nodes()
                if n is not None and n.get_type() == NodeType.FILE
            ]
            subgraph_id_dict[instance_id] = file_node_ids
            
        except Exception as e:
            print(f"❌ Error on {instance_id}: {e}")
            continue

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subgraph_id_dict, f, ensure_ascii=False, indent=2)

    print("✅ Done.")