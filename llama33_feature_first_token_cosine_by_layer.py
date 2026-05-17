#!/usr/bin/env python3
"""
Generate Llama-3.3-70B-Instruct responses, capture residual streams from every
transformer layer, and plot cosine similarity between one Goodfire layer-50 SAE
feature direction and the activation that predicts the first response token.

Example:
    uv run python llama33_feature_first_token_cosine_by_layer.py --direction both --feature-id 2734
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama33_say_nothing_sae_tokens import (
    CASE_ID,
    DEFAULT_EXAMPLES_PATH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_ID,
    NEURONPEDIA_LAYER,
    NEURONPEDIA_MODEL,
    NEURONPEDIA_RELEASE,
    SAE_FILENAME,
    SAE_LAYER,
    SAE_REPO_ID,
    GoodfireSparseAutoEncoder,
    PromptCase,
    first_parameter_device,
    load_goodfire_sae,
    load_model_and_tokenizer,
    load_prompt_cases,
    resolve_core_model,
    token_text,
)


DEFAULT_FEATURE_ID = 2734
DEFAULT_OUTPUT_DIR = Path("feature_2734_first_response_token_cosine_by_layer_llama33_70b_it_say_nothing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="4-bit Llama model id or local path.")
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-id", type=int, default=DEFAULT_FEATURE_ID)
    parser.add_argument(
        "--direction",
        choices=["original", "reversed", "both"],
        default="both",
        help="Which Say Nothing / Great Zoo prompt direction to run.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
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


def decoder_feature_vector(sae: GoodfireSparseAutoEncoder, feature_id: int) -> torch.Tensor:
    if feature_id < 0 or feature_id >= sae.d_hidden:
        raise ValueError(f"feature_id={feature_id} is outside SAE width {sae.d_hidden}.")
    return sae.decoder_linear.weight[:, feature_id].detach().float().cpu()


def capture_first_response_token_by_layer(
    case: PromptCase,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    args: argparse.Namespace,
) -> tuple[int, str, str, int, dict[int, torch.Tensor]]:
    core_model = resolve_core_model(model)
    input_device = torch.device(args.model_input_device) if args.model_input_device else first_parameter_device(model)
    inputs = tokenizer(case.prompt, return_tensors="pt").to(input_device)
    prompt_len = inputs["input_ids"].shape[-1]
    first_response_prediction_position = prompt_len - 1

    captured_resid: dict[int, list[torch.Tensor]] = {layer: [] for layer in range(len(core_model.layers))}
    hook_handles = []

    def make_hook(layer: int):
        def hook_fn(module, hook_input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured_resid[layer].append(hidden.detach().float().cpu())

        return hook_fn

    for layer_idx, layer in enumerate(core_model.layers):
        hook_handles.append(layer.register_forward_hook(make_hook(layer_idx)))

    generation_kwargs: dict[str, object] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    print(f"[{case.direction}] Generating response for: {case.question}")
    try:
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)
    finally:
        for handle in hook_handles:
            handle.remove()

    new_ids = output_ids[0][prompt_len:]
    if new_ids.numel() == 0:
        raise RuntimeError("Model generated no response tokens.")

    first_response_token_id = int(new_ids[0].item())
    first_response_token_text = token_text(tokenizer, first_response_token_id)
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    first_token_resid_by_layer: dict[int, torch.Tensor] = {}
    for layer_idx, chunks in captured_resid.items():
        if not chunks:
            raise RuntimeError(f"No residual activations were captured for layer {layer_idx}.")
        resid = torch.cat(chunks, dim=1).squeeze(0)
        first_token_resid_by_layer[layer_idx] = resid[first_response_prediction_position]

    return (
        first_response_token_id,
        first_response_token_text,
        generated_text,
        first_response_prediction_position,
        first_token_resid_by_layer,
    )


def cosine_by_layer(resid_by_layer: dict[int, torch.Tensor], feature_vector: torch.Tensor) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer_idx in sorted(resid_by_layer):
        resid_vector = resid_by_layer[layer_idx].float()
        cosine = F.cosine_similarity(resid_vector.unsqueeze(0), feature_vector.unsqueeze(0), dim=-1).item()
        rows.append({"layer": layer_idx, "cosine_similarity": float(cosine)})
    return rows


def write_layer_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "cosine_similarity"])
        writer.writeheader()
        writer.writerows(rows)


def plot_cosines(path: Path, case: PromptCase, feature_id: int, first_token_text: str, rows: list[dict[str, object]]) -> None:
    xs = [int(row["layer"]) for row in rows]
    ys = [float(row["cosine_similarity"]) for row in rows]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=180)
    ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.7)
    ax.plot(xs, ys, color="#1f77b4", linewidth=1.5)
    ax.scatter(xs, ys, color="#1f77b4", s=18)
    token_label = first_token_text.encode("unicode_escape").decode("ascii")
    ax.set_title(
        f"Feature {feature_id} Cosine Similarity by Layer ({case.direction}, first token: {token_label})",
        fontsize=11,
    )
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Cosine similarity")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_xticks(xs[::5] + ([xs[-1]] if xs[-1] % 5 else []))

    max_row = max(rows, key=lambda row: float(row["cosine_similarity"]))
    min_row = min(rows, key=lambda row: float(row["cosine_similarity"]))
    for row, label in [(max_row, "max"), (min_row, "min")]:
        x = int(row["layer"])
        y = float(row["cosine_similarity"])
        ax.annotate(
            f"{label} L{x}: {y:.3f}",
            xy=(x, y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": "#444444"},
        )

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_case(
    case: PromptCase,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    sae: GoodfireSparseAutoEncoder,
    args: argparse.Namespace,
) -> dict[str, object]:
    (
        first_token_id,
        first_token_text,
        generated_text,
        first_prediction_position,
        resid_by_layer,
    ) = capture_first_response_token_by_layer(case, tokenizer, model, args)
    feature_vector = decoder_feature_vector(sae, args.feature_id)
    rows = cosine_by_layer(resid_by_layer, feature_vector)

    case_dir = args.output_dir / case.direction
    case_dir.mkdir(parents=True, exist_ok=True)
    csv_path = case_dir / f"feature_{args.feature_id}_first_response_token_cosine_by_layer.csv"
    json_path = case_dir / f"feature_{args.feature_id}_first_response_token_cosine_by_layer.json"
    plot_path = case_dir / f"feature_{args.feature_id}_first_response_token_cosine_by_layer.png"

    write_layer_csv(csv_path, rows)
    result = {
        "model_id": args.model,
        "case_id": case.case_id,
        "category": case.category,
        "direction": case.direction,
        "truth": case.truth,
        "question": case.question,
        "prompt": case.prompt,
        "generated_text": generated_text,
        "analysis_type": "feature_decoder_cosine_first_response_token_by_layer",
        "feature_id": args.feature_id,
        "first_response_token_id": first_token_id,
        "first_response_token_text": first_token_text,
        "first_response_prediction_position": first_prediction_position,
        "sae_repo_id": SAE_REPO_ID,
        "sae_filename": SAE_FILENAME,
        "sae_layer": SAE_LAYER,
        "neuronpedia_model": NEURONPEDIA_MODEL,
        "neuronpedia_release": NEURONPEDIA_RELEASE,
        "neuronpedia_layer": NEURONPEDIA_LAYER,
        "neuronpedia_url": f"https://www.neuronpedia.org/{NEURONPEDIA_MODEL}/{NEURONPEDIA_LAYER}/{args.feature_id}",
        "layer_scores": rows,
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
    }
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_cosines(plot_path, case, args.feature_id, first_token_text, rows)

    print(f"[{case.direction}] Wrote plot: {plot_path}")
    return {
        "case_id": case.case_id,
        "direction": case.direction,
        "first_response_token_id": first_token_id,
        "first_response_token_text": first_token_text,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
        "max_cosine_similarity": max(float(row["cosine_similarity"]) for row in rows),
        "min_cosine_similarity": min(float(row["cosine_similarity"]) for row in rows),
    }


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

    summaries = [run_case(case, tokenizer, model, sae, args) for case in cases]
    summary = {
        "model_id": args.model,
        "examples_path": str(args.examples),
        "case_id": CASE_ID,
        "directions": [case.direction for case in cases],
        "max_new_tokens": args.max_new_tokens,
        "feature_id": args.feature_id,
        "seed": args.seed,
        "sae_repo_id": SAE_REPO_ID,
        "sae_filename": SAE_FILENAME,
        "sae_layer": SAE_LAYER,
        "sae_d_hidden": sae.d_hidden,
        "sae_expansion_factor": sae.d_hidden // d_model,
        "neuronpedia_model": NEURONPEDIA_MODEL,
        "neuronpedia_release": NEURONPEDIA_RELEASE,
        "neuronpedia_layer": NEURONPEDIA_LAYER,
        "cases": summaries,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
