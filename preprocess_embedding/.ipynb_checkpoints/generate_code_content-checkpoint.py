"""
Generate node content for all nodes in code graph
"""
from os.path import isfile
import os
import sys
import pandas as pd
import json
import tqdm
import re

# Ajouter le path du repo pour importer codegraph_parser
sys.path.append("CodeFuse-CGM/retriever")

from codegraph_parser.python.codegraph_python_local import parse, NodeType, EdgeType

def extract_code_and_doc(content):
    """
    Split code and doc
    """
    # match docstring
    docstring_pattern = r'"""(.*?)"""|\'\'\'(.*?)\'\'\''
    docstrings = re.findall(docstring_pattern, content, re.DOTALL)

    # extract pure code
    code_without_docstring = re.sub(docstring_pattern, '', content, flags=re.DOTALL)
    # merge docstring
    extracted_docstrings = "\n\n".join([d[0] or d[1] for d in docstrings])
    return code_without_docstring, extracted_docstrings

def get_graph_file_name(item):
    """
    Return graph_file_name
    """
    return item["graph_file"]

if __name__ == "__main__":

    # Lire le fichier JSON de base
    graph_basic_df = pd.read_json(
        "CodeFuse-CGM/preprocess_embedding/test_lite_basic_info.json",
        orient="index"
    )

    # Dossiers de données et output
    graph_data_path = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\CodeFuse-CGM-main\dataset\swe-bench-lite"
    node_content_path = "CodeFuse-CGM/retriever/content/output"

    # Créer le dossier output s'il n'existe pas
    os.makedirs(node_content_path, exist_ok=True)

    # Liste des graphes
    graph_list = os.listdir(graph_data_path)

    # Ajouter le chemin du fichier graph à la DataFrame
    graph_basic_df["graph_file"] = graph_basic_df.apply(lambda item: get_graph_file_name(item), axis=1)

    # Générer le contenu pour chaque repo
    for idx, item in tqdm.tqdm(graph_basic_df.iterrows(), total=len(graph_basic_df)):

        instance_id = item.instance_id
        graph_file = item.graph_file
        tmp_graph_data_path = os.path.join(graph_data_path, graph_file)

        # Chemin complet du fichier output
        output_file = os.path.join(node_content_path, f'{instance_id}.json')

        # Ignorer si déjà traité
        if os.path.isfile(output_file):
            continue

        # Charger le graphe avec encode UTF-8
        try:
            graph = parse(tmp_graph_data_path)
        except Exception as e:
            print(f"========= parse error: {tmp_graph_data_path} =========")
            print(e)
            continue

        try:
            nodes = graph.get_nodes()
        except Exception as e:
            print(f"========= get_nodes error: {tmp_graph_data_path} =========")
            print(e)
            continue

        node_code_dict = {}
        node_doc_dict = {}

        for node in nodes:
            node_id = node.node_id
            content = node.get_content()

            code, doc = extract_code_and_doc(content)

            node_code_dict[node_id] = code

            if doc.strip():
                node_doc_dict[node_id] = doc

        # Sauvegarder le résultat en UTF-8
        with open(output_file, 'w', encoding='utf-8') as json_file:
            node_content_dict = {
                "code": node_code_dict,
                "doc": node_doc_dict
            }
            json.dump(node_content_dict, json_file, ensure_ascii=False, indent=4)
