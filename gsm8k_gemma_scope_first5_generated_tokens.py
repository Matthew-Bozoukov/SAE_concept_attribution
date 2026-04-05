"""
Run a fixed question through Gemma 3 4B IT, capture residual streams for the
Gemma Scope 2 residual-stream SAEs, and save top SAE features for every
generated output token.

This version is intentionally simpler than the older GPT-OSS chain-of-thought
analysis flow: Gemma 3 4B IT is treated as a standard assistant response, so we
analyze each generated token position directly with no CoT span extraction.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests
import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-3-4b-it"
SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_LAYER_MAP = {
    9: {
        "sae_id": "layer_9_width_16k_l0_medium",
        "neuronpedia_layer": "9-gemmascope-2-res-16k",
    },
    17: {
        "sae_id": "layer_17_width_16k_l0_medium",
        "neuronpedia_layer": "17-gemmascope-2-res-16k",
    },
    22: {
        "sae_id": "layer_22_width_16k_l0_medium",
        "neuronpedia_layer": "22-gemmascope-2-res-16k",
    },
    29: {
        "sae_id": "layer_29_width_16k_l0_medium",
        "neuronpedia_layer": "29-gemmascope-2-res-16k",
    },
}
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TOP_K = 10
DEFAULT_OUTPUT_DIR = Path("top_sae_features_by_position_gemma_3_4b_it")

NEURONPEDIA_MODEL = "gemma-3-4b-it"
NEURONPEDIA_API_KEY = os.getenv("NEURONPEDIA_API_KEY", "")
NEURONPEDIA_TIMEOUT_SECONDS = 10

DEFAULT_QUESTION = (
    "How does your company's integration of new AI tools in customer service reflect "
    "your brand's commitment to innovation while maintaining the human touch that "
    "customers value? I'm interested in how these technologies enhance rather than "
    "replace the customer service experience your brand is known for."
)

SYSTEM_PROMPT = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Gemma 3 4B IT output-token activations with Gemma Scope 2 SAE features."
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question to send to the model.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top SAE features to save per output token.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where analysis outputs will be written.",
    )
    return parser.parse_args()


def layer_output_dir(output_dir: Path, layer: int) -> Path:
    return output_dir / f"layer_{layer}"


def fetch_feature_description(layer: int, feature_id: int) -> tuple[str | None, str | None]:
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
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        inner = model.language_model.model
        if hasattr(inner, "layers"):
            return inner
    raise ValueError("Unsupported model structure: could not find a transformer layer stack.")


def token_text(tokenizer: AutoTokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)


def generation_prediction_positions(prompt_len: int, num_generated_tokens: int) -> list[int]:
    if num_generated_tokens <= 0:
        return []
    return [prompt_len - 1 + idx for idx in range(num_generated_tokens)]


def top_features_for_resid_vector(
    sae: SAE,
    resid_vector: torch.Tensor,
    layer: int,
    top_k: int,
) -> list[dict[str, object]]:
    with torch.inference_mode():
        acts = sae.encode(resid_vector.unsqueeze(0)).squeeze(0)

    k = min(top_k, acts.shape[0])
    topk_vals, topk_idxs = torch.topk(acts, k)
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


def safe_token_preview(text: str, limit: int = 40) -> str:
    preview = text.encode("unicode_escape").decode("ascii")
    preview = preview.replace("\\", "_")
    preview = preview.replace("/", "_")
    preview = preview.replace(" ", "_")
    if not preview:
        preview = "empty"
    return preview[:limit]


def build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    summary_json_path = output_dir / "summary.json"

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

    prompt = build_prompt(tokenizer, args.question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    layers_to_capture = sorted(SAE_LAYER_MAP)
    captured_resid: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers_to_capture}
    hook_handles = []

    def make_hook(layer_idx: int):
        def hook_fn(module, hook_input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured_resid[layer_idx].append(hidden.detach().float())

        return hook_fn

    for layer in layers_to_capture:
        hook_handles.append(core_model.layers[layer].register_forward_hook(make_hook(layer)))

    print(f"Question: {args.question}\n")
    print("Generating response...")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    for hook_handle in hook_handles:
        hook_handle.remove()

    new_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    print(f"\nModel response:\n{response}\n")

    generated_token_ids = new_ids.tolist()
    n_generated_tokens = len(generated_token_ids)
    prompt_len = inputs["input_ids"].shape[-1]
    prediction_positions = generation_prediction_positions(prompt_len, n_generated_tokens)

    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "model_id": MODEL_ID,
        "sae_release": SAE_RELEASE,
        "question": args.question,
        "layers_analyzed": layers_to_capture,
        "top_k": args.top_k,
        "num_generated_tokens": int(n_generated_tokens),
        "neuronpedia_model": NEURONPEDIA_MODEL,
        "output_dir": str(output_dir),
        "generated_tokens": [
            {
                "generated_token_index": idx,
                "token_id": generated_token_ids[idx],
                "token_text": token_text(tokenizer, generated_token_ids[idx]),
                "prediction_position": prediction_positions[idx],
            }
            for idx in range(n_generated_tokens)
        ],
        "saved_files": [],
        "layers": {},
    }

    print("Computing per-layer SAE activations...")
    for layer in layers_to_capture:
        layer_key = str(layer)

        if not captured_resid[layer]:
            print(f"Layer {layer}: no activations captured.")
            summary["layers"][layer_key] = {"error": "no activations captured"}
            continue

        resid = torch.cat(captured_resid[layer], dim=1).squeeze(0)
        generated_resid = resid[prediction_positions]

        if generated_resid.shape[0] < n_generated_tokens:
            print(f"Layer {layer}: captured only {generated_resid.shape[0]} generated token activations.")

        sae_id = SAE_LAYER_MAP[layer]["sae_id"]
        print(f"Layer {layer}: loading SAE {SAE_RELEASE}/{sae_id}")

        try:
            sae = SAE.from_pretrained(
                release=SAE_RELEASE,
                sae_id=sae_id,
                device=str(device),
            )

            layer_dir = layer_output_dir(output_dir, layer)
            layer_dir.mkdir(parents=True, exist_ok=True)
            token_dir = layer_dir / "tokens"
            token_dir.mkdir(parents=True, exist_ok=True)

            token_output_paths = []
            for generated_token_index, token_id in enumerate(generated_token_ids):
                token_str = token_text(tokenizer, token_id)
                result = {
                    "model_id": MODEL_ID,
                    "sae_release": SAE_RELEASE,
                    "layer": layer,
                    "sae_id": sae_id,
                    "neuronpedia_layer": SAE_LAYER_MAP[layer]["neuronpedia_layer"],
                    "question": args.question,
                    "analysis_type": "token",
                    "generated_token_index": generated_token_index,
                    "token_id": token_id,
                    "token_text": token_str,
                    "prediction_position": prediction_positions[generated_token_index],
                    "top_features": top_features_for_resid_vector(
                        sae=sae,
                        resid_vector=generated_resid[generated_token_index],
                        layer=layer,
                        top_k=args.top_k,
                    ),
                }

                output_path = token_dir / (
                    f"gen_token_{generated_token_index:04d}_{safe_token_preview(token_str)}.json"
                )
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)

                token_output_paths.append(str(output_path))
                summary["saved_files"].append(str(output_path))

            summary["layers"][layer_key] = {
                "sae_id": sae_id,
                "neuronpedia_layer": SAE_LAYER_MAP[layer]["neuronpedia_layer"],
                "captured_generated_resid_shape": list(generated_resid.shape),
                "token_outputs": {
                    "num_saved_files": len(token_output_paths),
                    "output_files": token_output_paths,
                },
            }
        except Exception as e:
            print(f"Layer {layer}: error while processing SAE - {e}")
            summary["layers"][layer_key] = {"sae_id": sae_id, "error": str(e)}

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved per-token SAE features to: {output_dir}")
    print(f"Summary written to: {summary_json_path}")


if __name__ == "__main__":
    main()
