"""
Run a fixed math question through GPT-OSS-20B with chain-of-thought, capture
residual streams for the GPT-OSS SAE layers available in Neuronpedia /
`andyrdt/saes-gpt-oss-20b`, compute average SAE activations over prompt tokens,
and save top active features per layer to JSON.
"""

import json
import os
from typing import Any

import requests
import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "openai/gpt-oss-20b"
SAE_RELEASE = "gpt-oss-20b-andyrdt"
SAE_LAYER_MAP = {
    3: {
        "sae_id": "resid_post_layer_3_trainer_0",
        "neuronpedia_layer": "3-resid-post-aa",
    },
    7: {
        "sae_id": "resid_post_layer_7_trainer_0",
        "neuronpedia_layer": "7-resid-post-aa",
    },
    11: {
        "sae_id": "resid_post_layer_11_trainer_0",
        "neuronpedia_layer": "11-resid-post-aa",
    },
    15: {
        "sae_id": "resid_post_layer_15_trainer_0",
        "neuronpedia_layer": "15-resid-post-aa",
    },
    19: {
        "sae_id": "resid_post_layer_19_trainer_0",
        "neuronpedia_layer": "19-resid-post-aa",
    },
    23: {
        "sae_id": "resid_post_layer_23_trainer_0",
        "neuronpedia_layer": "23-resid-post-aa",
    },
}
MAX_NEW_TOKENS = 512
TOP_K = 10
OUTPUT_JSON_PATH = "top_sae_features_gpt_oss_20b.json"

# Neuronpedia config
NEURONPEDIA_MODEL = "gpt-oss-20b"
NEURONPEDIA_API_KEY = os.getenv("NEURONPEDIA_API_KEY", "")
NEURONPEDIA_TIMEOUT_SECONDS = 10

QUESTION = "Given x+y=10, find minimum of x^2+y^2."

COT_SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, showing your reasoning clearly. "
    "Think through each step before giving the final answer. "
    "End your response with 'The answer is: <number>'"
)


def fetch_feature_description(layer: int, feature_id: int) -> tuple[str | None, str | None]:
    """Return (description, error)."""
    neuronpedia_layer = SAE_LAYER_MAP[layer]["neuronpedia_layer"]
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


def resolve_core_model(model: AutoModelForCausalLM) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise ValueError("Unsupported model structure: expected model.model.layers")


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

core_model = resolve_core_model(model)
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
# Hook: capture residual stream at each configured SAE layer
# ---------------------------------------------------------------------------
layers_to_capture = sorted(SAE_LAYER_MAP)
captured_resid: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers_to_capture}
hook_handles = []


def make_hook(layer_idx: int):
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured_resid[layer_idx].append(hidden.detach().float())

    return hook_fn


for layer in layers_to_capture:
    hook_handles.append(core_model.layers[layer].register_forward_hook(make_hook(layer)))

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
    "layers_analyzed": layers_to_capture,
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

    resid = torch.cat(captured_resid[layer], dim=1).squeeze(0)
    resid = resid[:n_prompt_tokens]

    sae_id = SAE_LAYER_MAP[layer]["sae_id"]
    print(f"Layer {layer}: loading SAE {SAE_RELEASE}/{sae_id}")
    try:
        sae, cfg_dict, _ = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=sae_id,
        )
        sae = sae.to(device)
        with torch.inference_mode():
            sae_out = sae.encode(resid)

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
            "neuronpedia_layer": SAE_LAYER_MAP[layer]["neuronpedia_layer"],
            "captured_prompt_resid_shape": list(resid.shape),
            "top_features": top_features,
        }
    except Exception as e:
        print(f"Layer {layer}: error while processing SAE - {e}")
        results["layers"][layer_key] = {"sae_id": sae_id, "error": str(e)}

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved per-layer top-{TOP_K} SAE features to: {OUTPUT_JSON_PATH}")
