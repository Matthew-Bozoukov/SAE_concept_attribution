"""
Run a fixed math question through GPT-OSS-20B with chain-of-thought, capture
residual streams for the GPT-OSS SAE layers available in Neuronpedia /
`andyrdt/saes-gpt-oss-20b`, and save the top SAE features for each of the first
five generated tokens to JSON.
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
TOKENS_TO_ANALYZE = 5
TOP_K = 5
OUTPUT_JSON_PATH = "top_sae_features_first5_generated_tokens_gpt_oss_20b.json"

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


def token_text(tokenizer: AutoTokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)


def generation_prediction_positions(prompt_len: int, num_generated_tokens: int) -> list[int]:
    if num_generated_tokens <= 0:
        return []

    # The first generated token is predicted from the final prompt position.
    return [prompt_len - 1 + idx for idx in range(num_generated_tokens)]


def top_features_for_token(
    sae: SAE,
    resid_token: torch.Tensor,
    layer: int,
) -> list[dict[str, object]]:
    with torch.inference_mode():
        acts = sae.encode(resid_token.unsqueeze(0)).squeeze(0)

    topk_vals, topk_idxs = torch.topk(acts, TOP_K)
    top_features = []
    for rank, (feat_id, val) in enumerate(zip(topk_idxs.tolist(), topk_vals.tolist()), 1):
        feature_id = int(feat_id)
        description, description_error = fetch_feature_description(layer, feature_id)
        top_features.append(
            {
                "rank": rank,
                "feature_id": feature_id,
                "activation": float(val),
                "description": description,
                "description_error": description_error,
            }
        )
    return top_features


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

messages = [
    {"role": "system", "content": COT_SYSTEM_PROMPT},
    {"role": "user", "content": QUESTION},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

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

new_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
response = tokenizer.decode(new_ids, skip_special_tokens=True)
print(f"\nModel response:\n{response}\n")

n_generated_tokens = new_ids.shape[-1]
tokens_to_analyze = min(TOKENS_TO_ANALYZE, n_generated_tokens)
prompt_len = inputs["input_ids"].shape[-1]
prediction_positions = generation_prediction_positions(prompt_len, tokens_to_analyze)
results: dict[str, object] = {
    "model_id": MODEL_ID,
    "sae_release": SAE_RELEASE,
    "question": QUESTION,
    "layers_analyzed": layers_to_capture,
    "top_k": TOP_K,
    "tokens_to_analyze": tokens_to_analyze,
    "neuronpedia_model": NEURONPEDIA_MODEL,
    "generated_tokens": [
        {
            "generated_token_index": idx,
            "token_id": int(new_ids[idx].item()),
            "token_text": token_text(tokenizer, int(new_ids[idx].item())),
        }
        for idx in range(tokens_to_analyze)
    ],
    "layers": {},
}

print("Computing per-layer SAE activations for first generated tokens...")
for layer in layers_to_capture:
    layer_key = str(layer)

    if not captured_resid[layer]:
        print(f"Layer {layer}: no activations captured.")
        results["layers"][layer_key] = {"error": "no activations captured"}
        continue

    resid = torch.cat(captured_resid[layer], dim=1).squeeze(0)
    generated_resid = resid[prediction_positions]

    if generated_resid.shape[0] < tokens_to_analyze:
        print(
            f"Layer {layer}: captured only {generated_resid.shape[0]} generated token activations."
        )

    sae_id = SAE_LAYER_MAP[layer]["sae_id"]
    print(f"Layer {layer}: loading SAE {SAE_RELEASE}/{sae_id}")

    try:
        sae, cfg_dict, _ = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=sae_id,
        )
        sae = sae.to(device)

        token_results = []
        for token_idx in range(generated_resid.shape[0]):
            token_id = int(new_ids[token_idx].item())
            token_results.append(
                {
                    "generated_token_index": token_idx,
                    "token_id": token_id,
                    "token_text": token_text(tokenizer, token_id),
                    "prediction_position": prediction_positions[token_idx],
                    "top_features": top_features_for_token(
                        sae=sae,
                        resid_token=generated_resid[token_idx],
                        layer=layer,
                    ),
                }
            )

        results["layers"][layer_key] = {
            "sae_id": sae_id,
            "neuronpedia_layer": SAE_LAYER_MAP[layer]["neuronpedia_layer"],
            "captured_generated_resid_shape": list(generated_resid.shape),
            "tokens": token_results,
        }
    except Exception as e:
        print(f"Layer {layer}: error while processing SAE - {e}")
        results["layers"][layer_key] = {"sae_id": sae_id, "error": str(e)}

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(
    f"\nSaved per-layer top-{TOP_K} SAE features for the first {tokens_to_analyze} "
    f"generated tokens to: {OUTPUT_JSON_PATH}"
)
