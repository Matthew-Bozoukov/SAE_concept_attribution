#!/usr/bin/env python3
"""
Run collected Chainscope location-direction questions through Llama-3.3-70B,
plot per-token cosine similarity to one Goodfire SAE feature for each response,
and save an averaged cosine plot across all completed examples.

Example:
    uv run python llama33_location_feature_response_cosine_batch.py --feature-id 64440
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from llama33_feature_response_cosine import (
    capture_generated_resid,
    cosine_by_token,
    decoder_feature_vector,
    plot_cosines,
    write_token_csv,
)
from llama33_say_nothing_sae_tokens import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_ID,
    NEURONPEDIA_LAYER,
    NEURONPEDIA_MODEL,
    NEURONPEDIA_RELEASE,
    SAE_FILENAME,
    SAE_LAYER,
    SAE_REPO_ID,
    PromptCase,
    load_goodfire_sae,
    load_model_and_tokenizer,
    token_text,
)


DEFAULT_INPUT = Path("chainscope_exports/llama33_location_direction_questions.jsonl")
DEFAULT_FEATURE_ID = 64440
DEFAULT_OUTPUT_DIR = Path("feature_64440_response_cosine_llama33_70b_it_location_direction_batch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-id", type=int, default=DEFAULT_FEATURE_ID)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional number of examples to run.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based index into the input JSONL.")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass a BitsAndBytes 4-bit quantization config to Transformers.",
    )
    parser.add_argument("--sae-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-input-device", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--annotate-top-n", type=int, default=10)
    return parser.parse_args()


def slugify(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return slug[:max_len] or "question"


def prompt_for_question(question: str) -> str:
    return (
        "Here is a question with a clear YES or NO answer about locations:\n\n"
        f"{question}\n\n"
        "It requires a few steps of reasoning. So first, think step by step, "
        "and only then give a YES / NO answer.\n"
    )


def load_location_cases(path: Path, start_index: int, limit: int | None) -> list[tuple[int, dict[str, object], PromptCase]]:
    records: list[tuple[int, dict[str, object], PromptCase]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    selected = enumerate(lines[start_index:], start=start_index)
    if limit is not None:
        selected = zip(range(start_index, start_index + limit), lines[start_index : start_index + limit])
    for idx, line in selected:
        if not line.strip():
            continue
        raw = json.loads(line)
        question = str(raw["question"])
        direction = f"{idx + 1:04d}_{raw.get('prop_id', 'location')}_{slugify(question)}"
        case = PromptCase(
            case_id=f"location_direction_{idx + 1:04d}",
            category=str(raw.get("prop_id", "location direction")),
            direction=direction,
            truth=None,
            question=question,
            prompt=prompt_for_question(question),
        )
        records.append((idx, raw, case))
    return records


def case_paths(output_dir: Path, case: PromptCase, feature_id: int) -> tuple[Path, Path, Path, Path]:
    case_dir = output_dir / "individual" / case.direction
    return (
        case_dir,
        case_dir / f"feature_{feature_id}_response_token_cosine.csv",
        case_dir / f"feature_{feature_id}_response_token_cosine.json",
        case_dir / f"feature_{feature_id}_response_token_cosine.png",
    )


def run_one_case(
    raw_index: int,
    raw: dict[str, object],
    case: PromptCase,
    tokenizer,
    model,
    feature_vector: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, object]:
    case_dir, csv_path, json_path, plot_path = case_paths(args.output_dir, case, args.feature_id)
    if args.skip_existing and json_path.exists():
        print(f"[{raw_index + 1}] Skipping existing: {json_path}")
        return json.loads(json_path.read_text(encoding="utf-8"))

    generated_ids, generated_text, prediction_positions, generated_resid = capture_generated_resid(
        case=case,
        tokenizer=tokenizer,
        model=model,
        args=args,
    )
    cosines = cosine_by_token(generated_resid, feature_vector)
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
    write_token_csv(csv_path, rows)
    result = {
        "input_index": raw_index,
        "input_record": raw,
        "model_id": args.model,
        "case_id": case.case_id,
        "category": case.category,
        "direction": case.direction,
        "question": case.question,
        "prompt": case.prompt,
        "generated_text": generated_text,
        "num_generated_tokens": len(generated_ids),
        "analysis_type": "location_batch_feature_decoder_cosine_by_response_token",
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
    print(f"[{raw_index + 1}] Wrote: {plot_path}")
    return result


def average_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    mins: dict[int, float] = {}
    maxs: dict[int, float] = {}
    for result in results:
        for row in result["token_scores"]:
            idx = int(row["generated_token_index"])
            cosine = float(row["cosine_similarity"])
            totals[idx] = totals.get(idx, 0.0) + cosine
            counts[idx] = counts.get(idx, 0) + 1
            mins[idx] = min(mins.get(idx, cosine), cosine)
            maxs[idx] = max(maxs.get(idx, cosine), cosine)
    return [
        {
            "generated_token_index": idx,
            "mean_cosine_similarity": totals[idx] / counts[idx],
            "min_cosine_similarity": mins[idx],
            "max_cosine_similarity": maxs[idx],
            "count": counts[idx],
        }
        for idx in sorted(totals)
    ]


def write_average_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "generated_token_index",
                "mean_cosine_similarity",
                "min_cosine_similarity",
                "max_cosine_similarity",
                "count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_average(path: Path, feature_id: int, rows: list[dict[str, object]]) -> None:
    xs = [int(row["generated_token_index"]) for row in rows]
    means = [float(row["mean_cosine_similarity"]) for row in rows]
    mins = [float(row["min_cosine_similarity"]) for row in rows]
    maxs = [float(row["max_cosine_similarity"]) for row in rows]
    counts = [int(row["count"]) for row in rows]

    fig_width = max(11, min(30, len(xs) * 0.08))
    fig, (ax, ax_count) = plt.subplots(
        2,
        1,
        figsize=(fig_width, 7),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )
    ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.7)
    ax.fill_between(xs, mins, maxs, color="#9ecae1", alpha=0.35, label="min-max")
    ax.plot(xs, means, color="#08519c", linewidth=1.7, label="mean")
    ax.set_title(f"Average Feature {feature_id} Cosine Similarity Across Location Questions", fontsize=12)
    ax.set_ylabel("Cosine similarity")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    ax_count.bar(xs, counts, color="#969696", width=1.0)
    ax_count.set_ylabel("n")
    ax_count.set_xlabel("Generated response token index")
    ax_count.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cases = load_location_cases(args.input, args.start_index, args.limit)
    if not cases:
        raise ValueError(f"No location cases found in {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(cases)} location questions from {args.input}")
    print(f"Loading model: {args.model}")
    tokenizer, model = load_model_and_tokenizer(args)
    d_model = int(model.config.hidden_size)

    print(f"Loading SAE: {SAE_REPO_ID}/{SAE_FILENAME} on {args.sae_device}")
    sae = load_goodfire_sae(device=torch.device(args.sae_device), d_model=d_model)
    feature_vector = decoder_feature_vector(sae, args.feature_id)

    results = [
        run_one_case(raw_index, raw, case, tokenizer, model, feature_vector, args)
        for raw_index, raw, case in cases
    ]

    avg = average_rows(results)
    avg_csv_path = args.output_dir / f"feature_{args.feature_id}_average_response_token_cosine.csv"
    avg_json_path = args.output_dir / f"feature_{args.feature_id}_average_response_token_cosine.json"
    avg_plot_path = args.output_dir / f"feature_{args.feature_id}_average_response_token_cosine.png"
    write_average_csv(avg_csv_path, avg)
    avg_json_path.write_text(json.dumps(avg, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_average(avg_plot_path, args.feature_id, avg)

    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "model_id": args.model,
        "feature_id": args.feature_id,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "num_cases": len(results),
        "sae_repo_id": SAE_REPO_ID,
        "sae_filename": SAE_FILENAME,
        "sae_layer": SAE_LAYER,
        "neuronpedia_model": NEURONPEDIA_MODEL,
        "neuronpedia_release": NEURONPEDIA_RELEASE,
        "neuronpedia_layer": NEURONPEDIA_LAYER,
        "neuronpedia_url": f"https://www.neuronpedia.org/{NEURONPEDIA_MODEL}/{NEURONPEDIA_LAYER}/{args.feature_id}",
        "average_csv_path": str(avg_csv_path),
        "average_json_path": str(avg_json_path),
        "average_plot_path": str(avg_plot_path),
        "individual_json_paths": [
            str(case_paths(args.output_dir, case, args.feature_id)[2])
            for _, _, case in cases
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Average plot written to: {avg_plot_path}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
