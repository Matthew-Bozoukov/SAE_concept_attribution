#!/usr/bin/env python3
"""
Run the Say Nothing vs The Great Zoo of China prompt(s) through a 4-bit
Llama-3.3-70B-Instruct checkpoint, capture layer-50 residual streams, and save
the top active Neuronpedia/Goodfire SAE features for every generated token.

Neuronpedia API key:
    export NEURONPEDIA_API_KEY=...

Example:
    uv run python llama33_say_nothing_sae_tokens.py --direction both --max-new-tokens 512
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL_ID = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
DEFAULT_EXAMPLES_PATH = Path("llama31_70b_4bit_unfaithful_examples.json")
DEFAULT_OUTPUT_DIR = Path("top_sae_features_by_position_llama33_70b_it_say_nothing")
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TOP_K = 5

CASE_ID = "book_length_say_nothing_vs_great_zoo"

SAE_REPO_ID = "Goodfire/Llama-3.3-70B-Instruct-SAE-l50"
SAE_FILENAME = "Llama-3.3-70B-Instruct-SAE-l50.pt"
SAE_LAYER = 50

NEURONPEDIA_MODEL = "llama3.3-70b-it"
NEURONPEDIA_LAYER = "50-resid-post-gf"
NEURONPEDIA_RELEASE = "llama3.3-70b-it-gf"
NEURONPEDIA_API_KEY = os.getenv("NEURONPEDIA_API_KEY", "")
NEURONPEDIA_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class PromptCase:
    case_id: str
    category: str
    direction: str
    truth: str | None
    question: str
    prompt: str


class GoodfireSparseAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.encoder_linear = torch.nn.Linear(d_model, self.d_hidden)
        self.decoder_linear = torch.nn.Linear(self.d_hidden, d_model)
        self.to(dtype=dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_linear(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        return self.decode(features), features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="4-bit Llama model id or local path.")
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--direction",
        choices=["original", "reversed", "both"],
        default="both",
        help="Which Say Nothing / Great Zoo prompt direction to run.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Pass a BitsAndBytes 4-bit quantization config to Transformers. "
            "The default model is already a bnb-4bit checkpoint, so this is off by default."
        ),
    )
    parser.add_argument(
        "--sae-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the 4.3GB Goodfire SAE, e.g. cuda, cuda:1, or cpu.",
    )
    parser.add_argument(
        "--model-input-device",
        default=None,
        help="Optional device for tokenized inputs. Defaults to the first parameter device.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Transformers model/tokenizer loading.",
    )
    return parser.parse_args()


def load_prompt_cases(path: Path, direction: str) -> list[PromptCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    selected = [
        record
        for record in records
        if record.get("case_id") == CASE_ID
        and (direction == "both" or record.get("direction") == direction)
    ]
    if not selected:
        raise ValueError(f"No {CASE_ID!r} records found in {path} for direction={direction!r}.")
    return [
        PromptCase(
            case_id=str(record["case_id"]),
            category=str(record.get("category", "")),
            direction=str(record["direction"]),
            truth=record.get("truth"),
            question=str(record["question"]),
            prompt=str(record["prompt"]),
        )
        for record in selected
    ]


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


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def token_text(tokenizer: AutoTokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)


def safe_token_preview(text: str, limit: int = 40) -> str:
    preview = text.encode("unicode_escape").decode("ascii")
    for old, new in [("\\", "_"), ("/", "_"), (" ", "_")]:
        preview = preview.replace(old, new)
    return (preview or "empty")[:limit]


def generation_prediction_positions(prompt_len: int, num_generated_tokens: int) -> list[int]:
    return [prompt_len - 1 + idx for idx in range(num_generated_tokens)]


@lru_cache(maxsize=8192)
def fetch_feature_description(feature_id: int) -> tuple[str | None, str | None]:
    url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL}/{NEURONPEDIA_LAYER}/{feature_id}"
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
    except Exception as exc:
        return None, str(exc)


def load_goodfire_sae(device: torch.device, d_model: int) -> GoodfireSparseAutoEncoder:
    sae_path = hf_hub_download(repo_id=SAE_REPO_ID, filename=SAE_FILENAME)
    state_dict = torch.load(sae_path, weights_only=True, map_location="cpu")
    d_hidden, checkpoint_d_model = state_dict["encoder_linear.weight"].shape
    if checkpoint_d_model != d_model:
        raise ValueError(
            f"SAE checkpoint expects d_model={checkpoint_d_model}, but model config has d_model={d_model}."
        )
    sae = GoodfireSparseAutoEncoder(d_model=d_model, d_hidden=d_hidden, dtype=torch.bfloat16)
    sae.load_state_dict(state_dict)
    sae.to(device)
    sae.eval()
    return sae


def top_features_for_resid_vector(
    sae: GoodfireSparseAutoEncoder,
    resid_vector: torch.Tensor,
    top_k: int,
    sae_device: torch.device,
) -> list[dict[str, object]]:
    with torch.inference_mode():
        acts = sae.encode(resid_vector.to(device=sae_device, dtype=torch.bfloat16).unsqueeze(0)).squeeze(0)

    k = min(top_k, acts.shape[0])
    topk_vals, topk_idxs = torch.topk(acts.float(), k)
    top_features = []
    for rank, (feature_id, activation) in enumerate(zip(topk_idxs.tolist(), topk_vals.tolist()), 1):
        description, description_error = fetch_feature_description(int(feature_id))
        top_features.append(
            {
                "rank": rank,
                "feature_id": int(feature_id),
                "activation": float(activation),
                "description": description,
                "description_error": description_error,
                "neuronpedia_url": (
                    f"https://www.neuronpedia.org/{NEURONPEDIA_MODEL}/"
                    f"{NEURONPEDIA_LAYER}/{int(feature_id)}"
                ),
            }
        )
    return top_features


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model_kwargs: dict[str, object] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    return tokenizer, model


def run_case(
    case: PromptCase,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    sae: GoodfireSparseAutoEncoder,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, object]:
    core_model = resolve_core_model(model)
    if SAE_LAYER >= len(core_model.layers):
        raise ValueError(f"Model only has {len(core_model.layers)} layers; cannot hook layer {SAE_LAYER}.")

    input_device = torch.device(args.model_input_device) if args.model_input_device else first_parameter_device(model)
    inputs = tokenizer(case.prompt, return_tensors="pt").to(input_device)
    prompt_len = inputs["input_ids"].shape[-1]

    captured_resid: list[torch.Tensor] = []

    def hook_fn(module, hook_input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured_resid.append(hidden.detach().float().cpu())

    generation_kwargs: dict[str, object] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    hook_handle = core_model.layers[SAE_LAYER].register_forward_hook(hook_fn)
    print(f"[{case.direction}] Generating response for: {case.question}")
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                **generation_kwargs,
            )
    finally:
        hook_handle.remove()

    new_ids = output_ids[0][prompt_len:]
    generated_token_ids = new_ids.tolist()
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    prediction_positions = generation_prediction_positions(prompt_len, len(generated_token_ids))

    if not captured_resid:
        raise RuntimeError("No layer-50 residual activations were captured.")
    resid = torch.cat(captured_resid, dim=1).squeeze(0)
    generated_resid = resid[prediction_positions]

    case_dir = output_dir / case.direction
    token_dir = case_dir / "tokens"
    token_dir.mkdir(parents=True, exist_ok=True)

    token_outputs: list[str] = []
    sae_device = torch.device(args.sae_device)
    for generated_token_index, token_id in enumerate(generated_token_ids):
        text = token_text(tokenizer, token_id)
        result = {
            "model_id": args.model,
            "case_id": case.case_id,
            "direction": case.direction,
            "truth": case.truth,
            "question": case.question,
            "prompt": case.prompt,
            "analysis_type": "generated_token",
            "generated_token_index": generated_token_index,
            "token_id": int(token_id),
            "token_text": text,
            "prediction_position": prediction_positions[generated_token_index],
            "sae_repo_id": SAE_REPO_ID,
            "sae_filename": SAE_FILENAME,
            "sae_layer": SAE_LAYER,
            "neuronpedia_model": NEURONPEDIA_MODEL,
            "neuronpedia_release": NEURONPEDIA_RELEASE,
            "neuronpedia_layer": NEURONPEDIA_LAYER,
            "top_features": top_features_for_resid_vector(
                sae=sae,
                resid_vector=generated_resid[generated_token_index],
                top_k=args.top_k,
                sae_device=sae_device,
            ),
        }
        output_path = token_dir / f"gen_token_{generated_token_index:04d}_{safe_token_preview(text)}.json"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        token_outputs.append(str(output_path))

    case_summary = {
        "case_id": case.case_id,
        "category": case.category,
        "direction": case.direction,
        "truth": case.truth,
        "question": case.question,
        "prompt": case.prompt,
        "generated_text": generated_text,
        "num_generated_tokens": len(generated_token_ids),
        "captured_generated_resid_shape": list(generated_resid.shape),
        "generated_tokens": [
            {
                "generated_token_index": idx,
                "token_id": int(token_id),
                "token_text": token_text(tokenizer, int(token_id)),
                "prediction_position": prediction_positions[idx],
            }
            for idx, token_id in enumerate(generated_token_ids)
        ],
        "token_output_files": token_outputs,
    }
    (case_dir / "summary.json").write_text(
        json.dumps(case_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return case_summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cases = load_prompt_cases(args.examples, args.direction)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer, model = load_model_and_tokenizer(args)
    d_model = int(model.config.hidden_size)

    print(f"Loading SAE: {SAE_REPO_ID}/{SAE_FILENAME} on {args.sae_device}")
    sae = load_goodfire_sae(device=torch.device(args.sae_device), d_model=d_model)

    summaries = []
    for case in cases:
        summaries.append(
            run_case(
                case=case,
                tokenizer=tokenizer,
                model=model,
                sae=sae,
                args=args,
                output_dir=args.output_dir,
            )
        )

    summary = {
        "model_id": args.model,
        "examples_path": str(args.examples),
        "case_id": CASE_ID,
        "directions": [case.direction for case in cases],
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "seed": args.seed,
        "sae_repo_id": SAE_REPO_ID,
        "sae_filename": SAE_FILENAME,
        "sae_layer": SAE_LAYER,
        "sae_d_hidden": sae.d_hidden,
        "sae_expansion_factor": sae.d_hidden // d_model,
        "neuronpedia_model": NEURONPEDIA_MODEL,
        "neuronpedia_release": NEURONPEDIA_RELEASE,
        "neuronpedia_layer": NEURONPEDIA_LAYER,
        "neuronpedia_api_key_provided": bool(NEURONPEDIA_API_KEY),
        "cases": summaries,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved per-token SAE feature outputs to: {args.output_dir}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
