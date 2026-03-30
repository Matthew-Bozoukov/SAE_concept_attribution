"""
Run a fixed GSM8K question through Qwen/Qwen3-4B with chain-of-thought,
capture residual streams at layers 10-30, compute average SAE activations over
prompt tokens, and save top-10 active features per layer to JSON.

Question (Q4 from GSM8K test):
  James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint.
  How many total meters does he run a week?
"""

import json
import os
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-4B"
SAE_RELEASE = "mwhanna-qwen3-4b-transcoders"
LAYER_START = 10
LAYER_END = 30
MAX_NEW_TOKENS = 512
TOP_K = 10
OUTPUT_JSON_PATH = "top_sae_features_layers_10_30.json"

# Neuronpedia config
NEURONPEDIA_MODEL = "qwen3-4b"
NEURONPEDIA_API_KEY = os.getenv("NEURONPEDIA_API_KEY", "")
NEURONPEDIA_TIMEOUT_SECONDS = 10

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


def fetch_feature_description(layer: int, feature_id: int) -> tuple[str | None, str | None]:
    """Return (description, error)."""
    neuronpedia_layer = f"{layer}-transcoder-hp"
    url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL}/{neuronpedia_layer}/{feature_id}"
    headers = {"X-Api-Key": NEURONPEDIA_API_KEY} if NEURONPEDIA_API_KEY else None

    try:
        resp = requests.get(url, headers=headers, timeout=NEURONPEDIA_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        explanations = data.get("explanations", [])
        if not explanations:
            return None, "no explanation"

        description = explanations[0].get("description")
        if not description:
            return None, "empty description"
        return description, None
    except Exception as e:
        return None, str(e)


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

device = model.device
print(f"Model loaded on: {device}\n")

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
# Hook: capture residual stream at layers LAYER_START..LAYER_END
# ---------------------------------------------------------------------------
layers_to_capture = list(range(LAYER_START, LAYER_END + 1))
captured_resid: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers_to_capture}
hook_handles = []


def make_hook(layer_idx: int):
    def hook_fn(module, input, output):
        # output is (hidden_states, ...) for most transformer layers
        hidden = output[0] if isinstance(output, tuple) else output
        captured_resid[layer_idx].append(hidden.detach().float())

    return hook_fn


for layer in layers_to_capture:
    hook_handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))

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

for hook_handle in hook_handles:
    hook_handle.remove()

new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(new_ids, skip_special_tokens=True)
print(f"\nModel response:\n{response}\n")

# ---------------------------------------------------------------------------
# Compute SAE activations for each captured layer and save top-k to JSON
# ---------------------------------------------------------------------------
n_prompt_tokens = inputs["input_ids"].shape[-1]
results: dict[str, object] = {
    "model_id": MODEL_ID,
    "sae_release": SAE_RELEASE,
    "question": QUESTION,
    "layer_range": [LAYER_START, LAYER_END],
    "top_k": TOP_K,
    "neuronpedia_model": NEURONPEDIA_MODEL,
    "layers": {},
}

print("Computing per-layer SAE activations...")
for layer in layers_to_capture:
    layer_key = str(layer)

    if not captured_resid[layer]:
        print(f"Layer {layer}: no activations captured.")
        results["layers"][layer_key] = {"error": "no activations captured"}
        continue

    resid = torch.cat(captured_resid[layer], dim=1).squeeze(0)  # (total_tokens, d_model)
    resid = resid[:n_prompt_tokens]  # (n_prompt_tokens, d_model)

    sae_id = f"layer_{layer}"
    print(f"Layer {layer}: loading SAE {SAE_RELEASE}/{sae_id}")
    try:
        sae, cfg_dict, _ = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=sae_id,
        )
        sae = sae.to(device)
        with torch.inference_mode():
            sae_out = sae.encode(resid)  # (n_prompt_tokens, d_sae)

        mean_acts = sae_out.mean(dim=0)
        topk_vals, topk_idxs = torch.topk(mean_acts, TOP_K)

        top_features = []
        for rank, (feat_id, val) in enumerate(zip(topk_idxs.tolist(), topk_vals.tolist()), 1):
            feature_id = int(feat_id)
            description, description_error = fetch_feature_description(layer, feature_id)
            top_features.append(
                {
                    "rank": rank,
                    "feature_id": feature_id,
                    "mean_activation": float(val),
                    "description": description,
                    "description_error": description_error,
                }
            )

        results["layers"][layer_key] = {
            "sae_id": sae_id,
            "neuronpedia_layer": f"{layer}-transcoder-hp",
            "captured_prompt_resid_shape": list(resid.shape),
            "top_features": top_features,
        }
    except Exception as e:
        print(f"Layer {layer}: error while processing SAE - {e}")
        results["layers"][layer_key] = {"sae_id": sae_id, "error": str(e)}

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved per-layer top-{TOP_K} SAE features to: {OUTPUT_JSON_PATH}")
