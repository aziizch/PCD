import os
import json
import numpy as np

# --- CONFIGURATION ---
# 1. Your Reranker Results (Stage 2)
PREDICTION_DIR = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\reranker\reranker_outputs\stage_2_5\relevant_files"

# 2. The Official Answers
GROUND_TRUTH_PATH = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\ground_truth.json" 

def calculate_metrics(predictions, ground_truth, k=5):
    recalls = []
    precisions = []
    mrrs = []
    
    for instance_id, gold_files in ground_truth.items():
        # Get AI predictions (Top K)
        pred_files = predictions.get(instance_id, [])[:k]
        
        # Normalize paths (replace \ with /)
        gold_norm = {f.replace("\\", "/").strip() for f in gold_files}
        pred_norm = [f.replace("\\", "/").strip() for f in pred_files]
        
        # Calculate Hits
        hits = 0
        first_hit_rank = 0
        
        for rank, pred_file in enumerate(pred_norm, start=1):
            # Check if prediction is in ground truth
            # We check if the end of the path matches (to be safe against full/relative path mismatch)
            is_hit = False
            for gold in gold_norm:
                if pred_file.endswith(gold) or gold.endswith(pred_file):
                    is_hit = True
                    break
            
            if is_hit:
                hits += 1
                if first_hit_rank == 0:
                    first_hit_rank = rank
        
        # Recall: Hits / Total Correct Files
        if len(gold_files) > 0:
            recalls.append(hits / len(gold_files))
        else:
            recalls.append(0)
            
        # Precision: Hits / K
        precisions.append(hits / k)
        
        # MRR: 1 / Rank of first correct answer
        mrrs.append(1.0 / first_hit_rank if first_hit_rank > 0 else 0.0)
        
    return {
        f"Recall@{k}": np.mean(recalls),
        f"Precision@{k}": np.mean(precisions),
        f"MRR@{k}": np.mean(mrrs)
    }

if __name__ == "__main__":
    print("🚀 Starting Evaluation...")
    
    # Load Ground Truth
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"❌ Error: Ground Truth not found at {GROUND_TRUTH_PATH}")
        exit(1)
    with open(GROUND_TRUTH_PATH, "r") as f:
        ground_truth = json.load(f)

    # Load Predictions
    predictions = {}
    if not os.path.exists(PREDICTION_DIR):
        print(f"❌ Error: Prediction dir not found at {PREDICTION_DIR}")
        exit(1)
        
    print(f"📂 Loading predictions from: {PREDICTION_DIR}")
    for fname in os.listdir(PREDICTION_DIR):
        if not fname.endswith(".json"): continue
        instance_id = fname.replace(".json", "")
        
        with open(os.path.join(PREDICTION_DIR, fname), "r") as f:
            data = json.load(f)
            
        # Support both Stage 1 (list) and Stage 2 (dict) formats
        if isinstance(data, dict) and "selected" in data:
            predictions[instance_id] = data["selected"] # Stage 2
        elif isinstance(data, list):
            predictions[instance_id] = data # Stage 1
            
    print(f"✅ Loaded {len(predictions)} predictions.")
    print(f"✅ Loaded {len(ground_truth)} ground truth entries.")

    # Calculate
    metrics_1 = calculate_metrics(predictions, ground_truth, k=1)
    metrics_5 = calculate_metrics(predictions, ground_truth, k=5)
    
    print("\n" + "="*30)
    print("       EVALUATION RESULTS       ")
    print("="*30)
    print(f"Recall@1    : {metrics_1['Recall@1']:.2%}")
    print(f"Recall@5    : {metrics_5['Recall@5']:.2%}")
    print("-" * 30)
    print(f"Precision@1 : {metrics_1['Precision@1']:.2%}")
    print(f"Precision@5 : {metrics_5['Precision@5']:.2%}")
    print("-" * 30)
    print(f"MRR@5       : {metrics_5['MRR@5']:.4f}")
    print("="*30)