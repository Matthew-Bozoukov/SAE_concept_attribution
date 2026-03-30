"""
Run a fixed GSM8K question through google/gemma-3-4b-it with chain-of-thought,
hook the residual stream at layer 17, compute average SAE activations over all
tokens, and print the top-10 active features.

Question (Q4 from GSM8K test):
  James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint.
  How many total meters does he run a week?
"""

import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from sae_lens import SAE


MODEL_ID = "Qwen/Qwen3-4B"
SAE_RELEASE = "mwhanna-qwen3-4b-transcoders"
SAE_ID = "layer_8"
HOOK_LAYER = 8
MAX_NEW_TOKENS = 512
TOP_K = 10

# Neuronpedia config (model/layer derived from sae_lens pretrained_saes.yaml)
NEURONPEDIA_MODEL = "qwen3-4b"
NEURONPEDIA_LAYER = "8-transcoder-hp"
NEURONPEDIA_API_KEY = ""

QUESTION = (
    "James decides to run 3 sprints 3 times a week. "
    "He runs 60 meters each sprint. "
    "How many total meters does he run a week?"
)

COT_SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, showing your reasoning clearly. "
    "Think through each step before giving the final answer. "
    "End your response with 'The answer is: <number>'"
)

# ---------------------------------------------------------------------------
# Load model & tokenizer
# ---------------------------------------------------------------------------
print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print(model)
device = model.device
print(f"Model loaded on: {device}\n")

# ---------------------------------------------------------------------------
# Load SAE
# ---------------------------------------------------------------------------
print(f"Loading SAE: {SAE_RELEASE} / {SAE_ID}")
sae, cfg_dict, _ = SAE.from_pretrained(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
)
sae = sae.to(device)
print(f"SAE loaded. Hidden dim: {sae.cfg.d_sae}\n")

# ---------------------------------------------------------------------------
# Build prompt
# ---------------------------------------------------------------------------
messages = [
    {"role": "system", "content": COT_SYSTEM_PROMPT},
    {"role": "user", "content": QUESTION},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# ---------------------------------------------------------------------------
# Hook: capture residual stream at layer HOOK_LAYER
# ---------------------------------------------------------------------------
captured_resid: list[torch.Tensor] = []

def hook_fn(module, input, output):
    # output is (hidden_states, ...) for most transformer layers
    hidden = output[0] if isinstance(output, tuple) else output
    captured_resid.append(hidden.detach().float())

hook_handle = model.model.language_model.layers[HOOK_LAYER].register_forward_hook(hook_fn)

# ---------------------------------------------------------------------------
# Forward pass (generation)
# ---------------------------------------------------------------------------
print(f"Question: {QUESTION}\n")
print("Generating response...")

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

hook_handle.remove()

new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(new_ids, skip_special_tokens=True)
print(f"\nModel response:\n{response}\n")

# ---------------------------------------------------------------------------
# Compute SAE activations on the captured residual stream
# ---------------------------------------------------------------------------
# captured_resid may have multiple tensors if generation ran in chunks;
# concatenate along the token (sequence) dimension.
resid = torch.cat(captured_resid, dim=1)  # (1, total_tokens, d_model)
resid = resid.squeeze(0)                  # (total_tokens, d_model)

# Slice to prompt tokens only
n_prompt_tokens = inputs["input_ids"].shape[-1]
resid = resid[:n_prompt_tokens]           # (n_prompt_tokens, d_model)

print(f"Captured residual stream shape (prompt tokens only): {resid.shape}")
print("Computing SAE activations...")

with torch.inference_mode():
    sae_out = sae.encode(resid)  # (n_prompt_tokens, d_sae)

# Average over prompt tokens → (d_sae,)
mean_acts = sae_out.mean(dim=0)

#last_act=sae_out[-1]
# Top-K features
topk_vals, topk_idxs = torch.topk(mean_acts, TOP_K)

print(f"\nTop {TOP_K} SAE features (layer {HOOK_LAYER}, averaged over prompt tokens):")
print(f"{'Rank':>4}  {'Feature ID':>10}  {'Mean Activation':>16}")
print("-" * 36)
for rank, (feat_id, val) in enumerate(zip(topk_idxs.tolist(), topk_vals.tolist()), 1):
    print(f"{rank:>4}  {feat_id:>10}  {val:>16.4f}")

# ---------------------------------------------------------------------------
# Fetch feature explanations from Neuronpedia
# ---------------------------------------------------------------------------
print(f"\nFetching feature explanations from Neuronpedia...")
print(f"{'Rank':>4}  {'Feature ID':>10}  {'Mean Activation':>16}  {'Explanation'}")
print("-" * 90)

headers = {"X-Api-Key": NEURONPEDIA_API_KEY}
for rank, (feat_id, val) in enumerate(zip(topk_idxs.tolist(), topk_vals.tolist()), 1):
    url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL}/{NEURONPEDIA_LAYER}/{feat_id}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        explanations = data.get("explanations", [])
        explanation = explanations[0]["description"] if explanations else "(no explanation)"
    except Exception as e:
        explanation = f"(error: {e})"
    print(f"{rank:>4}  {feat_id:>10}  {val:>16.4f}  {explanation}")
