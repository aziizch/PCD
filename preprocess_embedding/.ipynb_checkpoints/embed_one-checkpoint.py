import os, json, pickle, argparse, time
import torch
from transformers import AutoModel, AutoConfig, Qwen2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--in_json", required=True)
parser.add_argument("--out_pkl", required=True)
parser.add_argument("--model_dir", required=True)
args = parser.parse_args()

BATCH = 4
MAX_SEQ_LEN = 256
MAX_CHARS = 8000

def clean_text(s):
    if s is None:
        return "x"
    s = str(s).strip()
    if len(s) == 0:
        return "x"
    return s[:MAX_CHARS]

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print(f"[worker] IN : {args.in_json}")
print(f"[worker] OUT: {args.out_pkl}")
print(f"[worker] device={device} dtype={dtype}")

t_load = time.time()
try:
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
except Exception:
    config = None

model = AutoModel.from_pretrained(
    args.model_dir,
    config=config,
    trust_remote_code=True,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
).to(device).eval()

# IMPORTANT: use Qwen2Tokenizer (slow), avoid cge-large tokenizer.model
tokenizer = Qwen2Tokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True,
    use_fast=False,
)

print(f"[worker] model+tokenizer loaded in {time.time()-t_load:.1f}s")

data = json.load(open(args.in_json, "r", encoding="utf-8"))
node_list = list((data.get("code") or {}).keys())
print(f"[worker] nodes={len(node_list)}")

node_emb = {"code": {}, "doc": {}}

t0 = time.time()
with torch.no_grad():
    for i in range(0, len(node_list), BATCH):
        batch_nodes = node_list[i:i+BATCH]
        batch_codes = [clean_text((data["code"].get(n) if "code" in data else None)) for n in batch_nodes]

        embs = model.encode(
            tokenizer,
            batch_codes,
            batch_size=len(batch_codes),
            show_progress_bar=False,
            max_seq_length=MAX_SEQ_LEN,
        )

        if torch.is_tensor(embs):
            embs = embs.detach().float().cpu().numpy()
            for j, n in enumerate(batch_nodes):
                node_emb["code"][n] = embs[j]
        elif isinstance(embs, list):
            for n, e in zip(batch_nodes, embs):
                node_emb["code"][n] = e.detach().float().cpu().numpy() if torch.is_tensor(e) else e
        else:
            for j, n in enumerate(batch_nodes):
                node_emb["code"][n] = embs[j]

        if (i // BATCH) % 50 == 0:
            done = min(i + BATCH, len(node_list))
            if device == "cuda":
                mem = torch.cuda.memory_allocated() / 1024**3
                print(f"[worker] {done}/{len(node_list)} | gpu_mem={mem:.2f}GB")
            else:
                print(f"[worker] {done}/{len(node_list)}")

with open(args.out_pkl, "wb") as f:
    pickle.dump(node_emb, f)

print(f"[worker] SAVED size={os.path.getsize(args.out_pkl)} bytes | time={time.time()-t0:.1f}s")