"""
Run a fixed math question through GPT-OSS-20B with chain-of-thought, capture
residual streams for the GPT-OSS SAE layers available in Neuronpedia /
`andyrdt/saes-gpt-oss-20b`, and save top SAE features for generated CoT content.

The script supports:
- token-level analysis for every CoT token position
- sentence-level analysis that averages residual activations across each sentence
  before passing the average through the SAE
"""

import argparse
import json
import os
import re
from pathlib import Path
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
DEFAULT_MAX_NEW_TOKENS = 792
DEFAULT_TOP_K = 10
DEFAULT_OUTPUT_DIR = Path("top_sae_features_by_position_gpt_oss_20b")

# Neuronpedia config
NEURONPEDIA_MODEL = "gpt-oss-20b"
NEURONPEDIA_API_KEY = os.getenv("NEURONPEDIA_API_KEY", "")
NEURONPEDIA_TIMEOUT_SECONDS = 10

DEFAULT_QUESTION = "Given x+y=10, find minimum of x^2+y^2."

COT_SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, showing your reasoning clearly. "
    "Think through each step before giving the final answer. "
    "End your response with 'The answer is: <number>'"
)


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])(?:\s+|$)|\n+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze GPT-OSS CoT activations with SAE features at token or sentence granularity."
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
        help="Number of top SAE features to save per analyzed unit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where analysis outputs will be written.",
    )
    parser.add_argument(
        "--analysis-mode",
        choices=["token", "sentence", "both"],
        default="token",
        help="Whether to save token-level outputs, sentence-level outputs, or both.",
    )
    return parser.parse_args()



def layer_output_dir(output_dir: Path, layer: int) -> Path:
    return output_dir / f"layer_{layer}"



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
    return [prompt_len - 1 + idx for idx in range(num_generated_tokens)]



def decoded_prefix_texts(tokenizer: AutoTokenizer, token_ids: list[int]) -> list[str]:
    prefixes: list[str] = []
    for end_idx in range(1, len(token_ids) + 1):
        prefixes.append(tokenizer.decode(token_ids[:end_idx], skip_special_tokens=False))
    return prefixes



def decoded_token_spans(tokenizer: AutoTokenizer, token_ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    if not token_ids:
        return "", []

    prefix_texts = decoded_prefix_texts(tokenizer, token_ids)
    spans: list[tuple[int, int]] = []
    prev_len = 0
    for prefix in prefix_texts:
        current_len = len(prefix)
        spans.append((prev_len, current_len))
        prev_len = current_len
    return prefix_texts[-1], spans



def first_token_ending_after(spans: list[tuple[int, int]], char_idx: int) -> int | None:
    for token_idx, (_, end) in enumerate(spans):
        if end > char_idx:
            return token_idx
    return None



def first_token_starting_at_or_after(spans: list[tuple[int, int]], char_idx: int) -> int | None:
    for token_idx, (start, _) in enumerate(spans):
        if start >= char_idx:
            return token_idx
    return None



def cot_token_span(tokenizer: AutoTokenizer, generated_token_ids: list[int]) -> tuple[int, int]:
    generated_text, spans = decoded_token_spans(tokenizer, generated_token_ids)
    normalized_text = generated_text.casefold()

    channel_pattern = re.compile(r"<\|channel\|>([^<]+)<\|message\|>", re.IGNORECASE)
    channels = list(channel_pattern.finditer(generated_text))
    analysis_match = next(
        (match for match in channels if match.group(1).strip().casefold() == "analysis"),
        None,
    )

    if analysis_match is not None:
        cot_start_char = analysis_match.end()
        end_match = next(
            (
                match
                for match in channels
                if match.start() >= cot_start_char
                and match.group(1).strip().casefold() in {"assistantfinal", "final"}
            ),
            None,
        )
        if end_match is None:
            tail_preview = generated_text[max(0, cot_start_char - 80) : cot_start_char + 400]
            raise ValueError(
                "Could not find final channel after analysis channel. "
                f"Decoded tail preview: {tail_preview!r}"
            )
        assistantfinal_char = end_match.start()
    else:
        analysis_char = normalized_text.find("analysis")
        if analysis_char == -1:
            raise ValueError("Could not find generated 'analysis' marker in model output.")

        cot_start_char = analysis_char + len("analysis")
        assistantfinal_char = normalized_text.find("assistantfinal", cot_start_char)
        if assistantfinal_char == -1:
            assistantfinal_char = normalized_text.find("final", cot_start_char)
        if assistantfinal_char == -1:
            tail_preview = generated_text[max(0, cot_start_char - 80) : cot_start_char + 400]
            raise ValueError(
                "Could not find generated final marker after 'analysis'. "
                f"Decoded tail preview: {tail_preview!r}"
            )

    cot_start = first_token_ending_after(spans, cot_start_char - 1)
    cot_end = first_token_starting_at_or_after(spans, assistantfinal_char)
    if cot_start is None or cot_end is None or cot_end <= cot_start:
        raise ValueError("The span between analysis and final output is empty.")

    return cot_start, cot_end



def split_cot_sentences(cot_text: str, token_spans: list[tuple[int, int]]) -> list[dict[str, object]]:
    sentences: list[dict[str, object]] = []
    start_char = 0

    for match in SENTENCE_SPLIT_PATTERN.finditer(cot_text):
        end_char = match.start()
        sentence_text = cot_text[start_char:end_char].strip()
        if sentence_text:
            sentences.append(sentence_record(cot_text, token_spans, start_char, end_char, sentence_text))
        start_char = match.end()

    trailing_text = cot_text[start_char:].strip()
    if trailing_text:
        sentences.append(sentence_record(cot_text, token_spans, start_char, len(cot_text), trailing_text))

    if not sentences and cot_text.strip():
        sentences.append(sentence_record(cot_text, token_spans, 0, len(cot_text), cot_text.strip()))

    return [sentence for sentence in sentences if sentence["token_start"] < sentence["token_end"]]



def sentence_record(
    cot_text: str,
    token_spans: list[tuple[int, int]],
    raw_start_char: int,
    raw_end_char: int,
    sentence_text: str,
) -> dict[str, object]:
    trimmed_start = cot_text.find(sentence_text, raw_start_char, raw_end_char)
    if trimmed_start == -1:
        trimmed_start = raw_start_char
    trimmed_end = trimmed_start + len(sentence_text)

    token_start = first_token_ending_after(token_spans, trimmed_start)
    token_end_inclusive = first_token_ending_after(token_spans, trimmed_end - 1)
    if token_start is None or token_end_inclusive is None:
        return {
            "text": sentence_text,
            "char_start": trimmed_start,
            "char_end": trimmed_end,
            "token_start": 0,
            "token_end": 0,
        }

    return {
        "text": sentence_text,
        "char_start": trimmed_start,
        "char_end": trimmed_end,
        "token_start": token_start,
        "token_end": token_end_inclusive + 1,
    }



def top_features_for_resid_vector(
    sae: SAE,
    resid_vector: torch.Tensor,
    layer: int,
    top_k: int,
) -> list[dict[str, object]]:
    with torch.inference_mode():
        acts = sae.encode(resid_vector.unsqueeze(0)).squeeze(0)

    topk_vals, topk_idxs = torch.topk(acts, top_k)
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

    messages = [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {"role": "user", "content": args.question},
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
    cot_start_idx, cot_end_idx = cot_token_span(tokenizer, generated_token_ids)
    selected_token_ids = generated_token_ids[cot_start_idx:cot_end_idx]
    selected_prediction_positions = prediction_positions[cot_start_idx:cot_end_idx]
    cot_text, cot_token_spans = decoded_token_spans(tokenizer, selected_token_ids)
    cot_sentences = split_cot_sentences(cot_text, cot_token_spans)

    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "model_id": MODEL_ID,
        "sae_release": SAE_RELEASE,
        "question": args.question,
        "layers_analyzed": layers_to_capture,
        "top_k": args.top_k,
        "analysis_mode": args.analysis_mode,
        "num_generated_tokens": int(n_generated_tokens),
        "num_cot_tokens": len(selected_token_ids),
        "num_cot_sentences": len(cot_sentences),
        "neuronpedia_model": NEURONPEDIA_MODEL,
        "output_dir": str(output_dir),
        "cot_span": {
            "start_generated_token_index": cot_start_idx,
            "end_generated_token_index_exclusive": cot_end_idx,
            "start_marker": "analysis",
            "end_marker": "assistantfinal",
        },
        "generated_tokens": [
            {
                "generated_token_index": idx,
                "token_id": generated_token_ids[idx],
                "token_text": token_text(tokenizer, generated_token_ids[idx]),
                "prediction_position": prediction_positions[idx],
                "in_cot_span": cot_start_idx <= idx < cot_end_idx,
            }
            for idx in range(n_generated_tokens)
        ],
        "cot_tokens": [
            {
                "cot_token_index": idx,
                "generated_token_index": cot_start_idx + idx,
                "token_id": token_id,
                "token_text": token_text(tokenizer, token_id),
                "prediction_position": selected_prediction_positions[idx],
            }
            for idx, token_id in enumerate(selected_token_ids)
        ],
        "cot_sentences": [
            {
                "sentence_index": idx,
                "text": sentence["text"],
                "token_start": sentence["token_start"],
                "token_end": sentence["token_end"],
                "generated_token_start": cot_start_idx + int(sentence["token_start"]),
                "generated_token_end_exclusive": cot_start_idx + int(sentence["token_end"]),
            }
            for idx, sentence in enumerate(cot_sentences)
        ],
        "saved_files": [],
        "layers": {},
    }

    do_token_analysis = args.analysis_mode in {"token", "both"}
    do_sentence_analysis = args.analysis_mode in {"sentence", "both"}

    print("Computing per-layer SAE activations...")
    for layer in layers_to_capture:
        layer_key = str(layer)

        if not captured_resid[layer]:
            print(f"Layer {layer}: no activations captured.")
            summary["layers"][layer_key] = {"error": "no activations captured"}
            continue

        resid = torch.cat(captured_resid[layer], dim=1).squeeze(0)
        generated_resid = resid[prediction_positions]
        cot_generated_resid = generated_resid[cot_start_idx:cot_end_idx]

        if generated_resid.shape[0] < n_generated_tokens:
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

            layer_dir = layer_output_dir(output_dir, layer)
            layer_dir.mkdir(parents=True, exist_ok=True)
            layer_summary: dict[str, object] = {
                "sae_id": sae_id,
                "neuronpedia_layer": SAE_LAYER_MAP[layer]["neuronpedia_layer"],
                "captured_generated_resid_shape": list(generated_resid.shape),
                "captured_cot_resid_shape": list(cot_generated_resid.shape),
            }

            if do_token_analysis:
                token_dir = layer_dir / "tokens"
                token_dir.mkdir(parents=True, exist_ok=True)
                token_output_paths = []
                for cot_token_idx in range(cot_generated_resid.shape[0]):
                    generated_token_index = cot_start_idx + cot_token_idx
                    token_id = selected_token_ids[cot_token_idx]
                    token_str = token_text(tokenizer, token_id)
                    result = {
                        "model_id": MODEL_ID,
                        "sae_release": SAE_RELEASE,
                        "layer": layer,
                        "sae_id": sae_id,
                        "neuronpedia_layer": SAE_LAYER_MAP[layer]["neuronpedia_layer"],
                        "question": args.question,
                        "analysis_type": "token",
                        "cot_token_index": cot_token_idx,
                        "generated_token_index": generated_token_index,
                        "token_id": token_id,
                        "token_text": token_str,
                        "prediction_position": selected_prediction_positions[cot_token_idx],
                        "top_features": top_features_for_resid_vector(
                            sae=sae,
                            resid_vector=cot_generated_resid[cot_token_idx],
                            layer=layer,
                            top_k=args.top_k,
                        ),
                    }

                    output_path = token_dir / (
                        f"cot_token_{cot_token_idx:04d}_gen_{generated_token_index:04d}_{safe_token_preview(token_str)}.json"
                    )
                    with output_path.open("w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2)

                    token_output_paths.append(str(output_path))
                    summary["saved_files"].append(str(output_path))

                layer_summary["token_outputs"] = {
                    "num_saved_files": len(token_output_paths),
                    "output_files": token_output_paths,
                }

            if do_sentence_analysis:
                sentence_dir = layer_dir / "sentences"
                sentence_dir.mkdir(parents=True, exist_ok=True)
                sentence_output_paths = []
                for sentence_idx, sentence in enumerate(cot_sentences):
                    token_start = int(sentence["token_start"])
                    token_end = int(sentence["token_end"])
                    sentence_resid = cot_generated_resid[token_start:token_end]
                    if sentence_resid.shape[0] == 0:
                        continue

                    mean_sentence_resid = sentence_resid.mean(dim=0)
                    generated_token_start = cot_start_idx + token_start
                    generated_token_end = cot_start_idx + token_end
                    result = {
                        "model_id": MODEL_ID,
                        "sae_release": SAE_RELEASE,
                        "layer": layer,
                        "sae_id": sae_id,
                        "neuronpedia_layer": SAE_LAYER_MAP[layer]["neuronpedia_layer"],
                        "question": args.question,
                        "analysis_type": "sentence",
                        "sentence_index": sentence_idx,
                        "sentence_text": sentence["text"],
                        "cot_token_start": token_start,
                        "cot_token_end_exclusive": token_end,
                        "generated_token_start": generated_token_start,
                        "generated_token_end_exclusive": generated_token_end,
                        "num_tokens_averaged": int(sentence_resid.shape[0]),
                        "top_features": top_features_for_resid_vector(
                            sae=sae,
                            resid_vector=mean_sentence_resid,
                            layer=layer,
                            top_k=args.top_k,
                        ),
                    }

                    output_path = sentence_dir / (
                        f"sentence_{sentence_idx:04d}_gen_{generated_token_start:04d}_{generated_token_end:04d}.json"
                    )
                    with output_path.open("w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2)

                    sentence_output_paths.append(str(output_path))
                    summary["saved_files"].append(str(output_path))

                layer_summary["sentence_outputs"] = {
                    "num_saved_files": len(sentence_output_paths),
                    "output_files": sentence_output_paths,
                }

            summary["layers"][layer_key] = layer_summary
        except Exception as e:
            print(f"Layer {layer}: error while processing SAE - {e}")
            summary["layers"][layer_key] = {"sae_id": sae_id, "error": str(e)}

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\nSaved SAE features for analysis mode '{args.analysis_mode}' to: {output_dir}"
    )
    print(f"Summary written to: {summary_json_path}")


if __name__ == "__main__":
    main()
