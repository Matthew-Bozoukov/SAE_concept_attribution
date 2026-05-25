#!/usr/bin/env python3
"""
Generate Llama-3.3-70B-Instruct responses, capture layer-50 residual streams,
and plot cosine similarity between one Goodfire SAE feature direction and each
generated response-token residual.

Example:
    uv run python llama33_feature_response_cosine.py --direction both --feature-id 2734
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import replace
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
    generation_prediction_positions,
    load_goodfire_sae,
    load_model_and_tokenizer,
    load_prompt_cases,
    resolve_core_model,
    token_text,
)


DEFAULT_FEATURE_ID = 2734
DEFAULT_OUTPUT_DIR = Path("feature_2734_response_cosine_llama33_70b_it_say_nothing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="4-bit Llama model id or local path.")
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES_PATH)
    parser.add_argument(
        "--cases-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL file of arbitrary PromptCase rows. Each line should include "
            "case_id, category, direction, question, and prompt; truth is optional. "
            "When provided, --examples and --direction are ignored."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional .txt file whose contents replace the prompt loaded from --examples.",
    )
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
    parser.add_argument(
        "--annotate-top-n",
        type=int,
        default=10,
        help="Number of highest-cosine tokens to annotate in the plot.",
    )
    return parser.parse_args()


def safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "case"


def apply_prompt_file(cases: list[PromptCase], prompt_file: Path | None) -> list[PromptCase]:
    if prompt_file is None:
        return cases
    prompt = prompt_file.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    return [replace(case, prompt=prompt) for case in cases]


def load_cases_jsonl(path: Path) -> list[PromptCase]:
    cases: list[PromptCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        try:
            case = PromptCase(
                case_id=str(record["case_id"]),
                category=str(record.get("category", "")),
                direction=str(record["direction"]),
                truth=record.get("truth"),
                question=str(record["question"]),
                prompt=str(record["prompt"]),
            )
        except KeyError as exc:
            raise ValueError(f"{path}:{line_number} is missing required field {exc}.") from exc
        cases.append(case)
    if not cases:
        raise ValueError(f"No prompt cases found in {path}.")
    return cases


def decoder_feature_vector(sae: GoodfireSparseAutoEncoder, feature_id: int) -> torch.Tensor:
    if feature_id < 0 or feature_id >= sae.d_hidden:
        raise ValueError(f"feature_id={feature_id} is outside SAE width {sae.d_hidden}.")
    return sae.decoder_linear.weight[:, feature_id].detach().float().cpu()


def capture_generated_resid(
    case: PromptCase,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    args: argparse.Namespace,
) -> tuple[list[int], str, list[int], torch.Tensor]:
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
            output_ids = model.generate(**inputs, **generation_kwargs)
    finally:
        hook_handle.remove()

    generated_ids = output_ids[0][prompt_len:].tolist()
    generated_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    prediction_positions = generation_prediction_positions(prompt_len, len(generated_ids))

    if not captured_resid:
        raise RuntimeError("No layer-50 residual activations were captured.")

    resid = torch.cat(captured_resid, dim=1).squeeze(0)
    return generated_ids, generated_text, prediction_positions, resid[prediction_positions]


def cosine_by_token(resid_vectors: torch.Tensor, feature_vector: torch.Tensor) -> list[float]:
    if resid_vectors.shape[0] == 0:
        raise ValueError("Cannot plot cosine similarity for an empty response.")
    feature_vector = feature_vector.to(dtype=resid_vectors.dtype).unsqueeze(0)
    return F.cosine_similarity(resid_vectors, feature_vector, dim=-1).tolist()


def write_token_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "generated_token_index",
                "token_id",
                "token_text",
                "prediction_position",
                "cosine_similarity",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_cosines(
    path: Path,
    case: PromptCase,
    feature_id: int,
    rows: list[dict[str, object]],
    annotate_top_n: int,
) -> None:
    xs = [int(row["generated_token_index"]) for row in rows]
    ys = [float(row["cosine_similarity"]) for row in rows]

    fig_width = max(10, min(28, len(xs) * 0.18))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5), dpi=180)
    ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.7)
    ax.plot(xs, ys, color="#1f77b4", linewidth=1.5)
    ax.scatter(xs, ys, color="#1f77b4", s=12)
    ax.set_title(
        f"Feature {feature_id} Cosine Similarity vs Response Tokens ({case.direction})",
        fontsize=12,
    )
    ax.set_xlabel("Generated response token index")
    ax.set_ylabel("Cosine similarity with layer-50 residual")
    ax.grid(True, axis="y", alpha=0.25)

    if len(xs) <= 80:
        labels = [str(row["token_text"]).encode("unicode_escape").decode("ascii") for row in rows]
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)

    top_rows = sorted(rows, key=lambda row: float(row["cosine_similarity"]), reverse=True)[:annotate_top_n]
    for row in top_rows:
        x = int(row["generated_token_index"])
        y = float(row["cosine_similarity"])
        label = str(row["token_text"]).encode("unicode_escape").decode("ascii")[:18]
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            rotation=45,
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
    generated_ids, generated_text, prediction_positions, generated_resid = capture_generated_resid(
        case=case,
        tokenizer=tokenizer,
        model=model,
        args=args,
    )
    feature_vector = decoder_feature_vector(sae, args.feature_id)
    cosines = cosine_by_token(generated_resid, feature_vector)

    case_dir = args.output_dir / safe_path_component(f"{case.direction}_{case.case_id}")
    case_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "generated_token_index": idx,
            "token_id": int(token_id),
            "token_text": token_text(tokenizer, int(token_id)),
            "prediction_position": prediction_positions[idx],
            "cosine_similarity": float(cosines[idx]),
        }
        for idx, token_id in enumerate(generated_ids)
    ]

    csv_path = case_dir / f"feature_{args.feature_id}_response_token_cosine.csv"
    json_path = case_dir / f"feature_{args.feature_id}_response_token_cosine.json"
    plot_path = case_dir / f"feature_{args.feature_id}_response_token_cosine.png"

    write_token_csv(csv_path, rows)
    result = {
        "model_id": args.model,
        "case_id": case.case_id,
        "category": case.category,
        "direction": case.direction,
        "truth": case.truth,
        "question": case.question,
        "prompt": case.prompt,
        "generated_text": generated_text,
        "num_generated_tokens": len(generated_ids),
        "analysis_type": "feature_decoder_cosine_by_response_token",
        "feature_id": args.feature_id,
        "sae_repo_id": SAE_REPO_ID,
        "sae_filename": SAE_FILENAME,
        "sae_layer": SAE_LAYER,
        "neuronpedia_model": NEURONPEDIA_MODEL,
        "neuronpedia_release": NEURONPEDIA_RELEASE,
        "neuronpedia_layer": NEURONPEDIA_LAYER,
        "neuronpedia_url": f"https://www.neuronpedia.org/{NEURONPEDIA_MODEL}/{NEURONPEDIA_LAYER}/{args.feature_id}",
        "token_scores": rows,
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
    }
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_cosines(plot_path, case, args.feature_id, rows, args.annotate_top_n)

    print(f"[{case.direction}] Wrote plot: {plot_path}")
    return {
        "case_id": case.case_id,
        "direction": case.direction,
        "num_generated_tokens": len(generated_ids),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
        "max_cosine_similarity": max(cosines),
        "min_cosine_similarity": min(cosines),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.cases_jsonl is not None:
        cases = load_cases_jsonl(args.cases_jsonl)
        if args.prompt_file is not None:
            raise ValueError("--prompt-file cannot be combined with --cases-jsonl.")
    else:
        cases = apply_prompt_file(load_prompt_cases(args.examples, args.direction), args.prompt_file)
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
        "cases_jsonl": str(args.cases_jsonl) if args.cases_jsonl else None,
        "prompt_file": str(args.prompt_file) if args.prompt_file else None,
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
