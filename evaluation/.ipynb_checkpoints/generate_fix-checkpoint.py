
"""
Générateur de Correctif (Démonstration)
"""
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- CONFIG ---
MODEL_PATH = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\Qwen"
# On prend un bug qu'on sait que vous avez trouvé
TARGET_BUG_ID = "psf__requests-2317" 
RETRIEVED_FILES = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\reranker\reranker_outputs\stage_2_5\relevant_files"
SUBGRAPH_DIR = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\retriever\subgraphs_final"

# --- PROMPT ---
PROMPT_TEMPLATE = """You are a software engineer.
Issue: {issue}

I have identified the relevant file: {filename}
Here is its content:
{content}

Please write a code patch (diff) to fix this issue.
"""

# --- MODEL ---
print("🚀 Loading Model...")
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", quantization_config=quantization_config, trust_remote_code=True)

# --- LOAD DATA ---
# 1. Lire les résultats du reranker pour ce bug
with open(os.path.join(RETRIEVED_FILES, TARGET_BUG_ID + ".json"), "r") as f:
    data = json.load(f)
    best_file = data["selected"][0] # Le meilleur fichier trouvé

print(f"🎯 Fichier cible : {best_file}")

# 2. Lire la description du bug (il faut recharger basic_info, je simplifie ici)
# Idéalement, copiez-collez la description du bug ici pour le test
ISSUE_DESC = "The separability_matrix function does not work correctly with nested CompoundModels."

# 3. Récupérer le contenu du fichier (Simplifié: on suppose qu'on l'a)
# Dans la vraie vie, il faudrait parser le graphe comme dans reranker.py
FILE_CONTENT = "# (Contenu simulé pour la démo)... def separability_matrix(): ..." 

# --- GENERATION ---
prompt = PROMPT_TEMPLATE.format(issue=ISSUE_DESC, filename=best_file, content=FILE_CONTENT)
inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], return_tensors="pt", tokenize=True, add_generation_prompt=True)
inputs = inputs.to("cuda")

print("🤖 Génération du correctif...")
outputs = model.generate(inputs, max_new_tokens=512, do_sample=False)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== CORRECTIF PROPOSÉ PAR L'IA ===")
print(text)