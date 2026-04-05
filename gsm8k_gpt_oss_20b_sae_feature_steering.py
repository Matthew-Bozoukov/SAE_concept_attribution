"""
Steer GPT-OSS-20B on the GSM8K-style question used elsewhere in this repo by
adding a chosen SAE feature decoder vector at a specific transformer layer.

This script mirrors the prompt/question setup from
`gsm8k_gemma_scope_first5_generated_tokens.py`, but instead of analyzing
generated activations it intervenes during generation.

For now, steering is applied only at the SAE layer supplied by the user.
Token selection can be specified either in prompt-token indices or in
generated-token indices, where generated index 0 is the first token after the
prompt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "openai/gpt-oss-20b"
SAE_RELEASE = "gpt-oss-20b-andyrdt"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_STRENGTH = 10.0
DEFAULT_OUTPUT_PATH = Path("gpt_oss_20b_sae_feature_steering_output.json")
DEFAULT_QUESTION = "Given x+y=10, find minimum of x^2+y^2."

COT_SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, showing your reasoning clearly. "
    "Think through each step before giving the final answer. "
    "End your response with 'The answer is: <number>'"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question to send to the model.")
    parser.add_argument(
        "--system-prompt",
        default=COT_SYSTEM_PROMPT,
        help="System prompt used to build the chat template.",
    )
    parser.add_argument(
        "--sae-layer",
        type=int,
        required=True,
        help="Transformer layer whose resid-post SAE should be loaded and steered.",
    )
    parser.add_argument(
        "--feature-id",
        type=int,
        required=True,
        help="SAE feature id whose decoder vector will be used for steering.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=DEFAULT_STRENGTH,
        help="Steering coefficient applied to the SAE decoder vector.",
    )
    parser.add_argument(
        "--steer-tokens",
        default="",
        help="Comma-separated generated-token indices and/or ranges to steer, e.g. '0,1,3-5'.",
    )
    parser.add_argument(
        "--steer-prompt-tokens",
        default="",
        help=(
            "Comma-separated prompt-token indices and/or ranges to steer. "
            "Supports negative indices relative to the end of the prompt, e.g. '-5--1'."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Optional JSON output path for metadata and generations.",
    )
    return parser.parse_args()


def parse_token_indices(raw: str) -> list[int]:
    tokens: set[int] = set()
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start_str, end_str = item.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid token range {item!r}: end must be >= start.")
            tokens.update(range(start, end + 1))
        else:
            tokens.add(int(item))

    if not tokens:
        return []
    if min(tokens) < 0:
        raise ValueError("--steer-tokens cannot contain negative indices.")
    return sorted(tokens)


def parse_prompt_token_indices(raw: str, prompt_len: int) -> list[int]:
    if not raw.strip():
        return []

    tokens: set[int] = set()
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue

        if "-" in item[1:]:
            split_at = item[1:].find("-") + 1
            start = int(item[:split_at])
            end = int(item[split_at + 1 :])
            if end < start:
                raise ValueError(f"Invalid prompt-token range {item!r}: end must be >= start.")
            raw_indices = range(start, end + 1)
        else:
            raw_indices = [int(item)]

        for idx in raw_indices:
            resolved_idx = prompt_len + idx if idx < 0 else idx
            if not 0 <= resolved_idx < prompt_len:
                raise ValueError(
                    f"Prompt token index {idx} resolves to {resolved_idx}, "
                    f"which is outside [0, {prompt_len - 1}]."
                )
            tokens.add(resolved_idx)

    return sorted(tokens)


def sae_id_for_layer(layer: int) -> str:
    return f"resid_post_layer_{layer}_trainer_0"


def resolve_core_model(model: AutoModelForCausalLM) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise ValueError("Unsupported model structure: expected model.model.layers")


def build_inputs(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    question: str,
) -> tuple[str, dict[str, torch.Tensor]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    return prompt, inputs


class AbsolutePositionSteeringHook:
    def __init__(
        self,
        vector: torch.Tensor,
        strength: float,
        absolute_positions: set[int],
    ) -> None:
        self.vector = vector
        self.strength = strength
        self.absolute_positions = absolute_positions

    def __call__(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()

        seq_len = hidden.shape[1]
        matched_positions = [pos for pos in range(seq_len) if pos in self.absolute_positions]
        if matched_positions:
            steering = (self.strength * self.vector).to(device=hidden.device, dtype=hidden.dtype)
            hidden[:, matched_positions, :] += steering.view(1, 1, -1)

        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: dict[str, torch.Tensor],
    hook_layer: int | None = None,
    hook=None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> tuple[torch.Tensor, str]:
    core_model = resolve_core_model(model)
    handle = None
    if hook is not None:
        handle = core_model.layers[hook_layer].register_forward_hook(hook)

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **{key: value.to(model.device) for key, value in inputs.items()},
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
    finally:
        if handle is not None:
            handle.remove()

    prompt_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[0, prompt_len:].detach().cpu()
    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    return new_ids, response


def main() -> None:
    args = parse_args()
    steer_token_indices = parse_token_indices(args.steer_tokens)

    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model loaded on: {model.device}")

    prompt, inputs = build_inputs(
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
        question=args.question,
    )
    prompt_len = inputs["input_ids"].shape[1]
    steer_prompt_token_indices = parse_prompt_token_indices(args.steer_prompt_tokens, prompt_len)
    target_prediction_positions = {prompt_len - 1 + idx for idx in steer_token_indices}
    target_prompt_positions = set(steer_prompt_token_indices)
    target_positions = target_prompt_positions | target_prediction_positions
    if not target_positions:
        raise ValueError("Specify at least one token to steer via --steer-prompt-tokens and/or --steer-tokens.")

    sae_id = sae_id_for_layer(args.sae_layer)
    print(f"Loading SAE: {SAE_RELEASE}/{sae_id}")
    sae, cfg_dict, _ = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=sae_id,
    )
    sae = sae.to(model.device)
    if not 0 <= args.feature_id < sae.W_dec.shape[0]:
        raise ValueError(
            f"--feature-id must be in [0, {sae.W_dec.shape[0] - 1}] for {sae_id}, got {args.feature_id}."
        )

    feature_vector = sae.W_dec[args.feature_id].detach().to(model.device)

    print("Running baseline generation...")
    baseline_ids, baseline_response = generate_response(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        max_new_tokens=args.max_new_tokens,
    )

    steering_hook = AbsolutePositionSteeringHook(
        vector=feature_vector,
        strength=args.strength,
        absolute_positions=target_positions,
    )

    print("Running steered generation...")
    steered_ids, steered_response = generate_response(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        hook_layer=args.sae_layer,
        hook=steering_hook,
        max_new_tokens=args.max_new_tokens,
    )

    result = {
        "model_id": MODEL_ID,
        "sae_release": SAE_RELEASE,
        "sae_id": sae_id,
        "sae_cfg": cfg_dict,
        "question": args.question,
        "system_prompt": args.system_prompt,
        "prompt": prompt,
        "prompt_length_tokens": prompt_len,
        "sae_layer": args.sae_layer,
        "feature_id": args.feature_id,
        "strength": args.strength,
        "steer_prompt_token_indices": steer_prompt_token_indices,
        "steer_generated_token_indices": steer_token_indices,
        "steer_prompt_positions": sorted(target_prompt_positions),
        "steer_prediction_positions": sorted(target_prediction_positions),
        "steer_absolute_positions": sorted(target_positions),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "use_cache": False,
            "do_sample": False,
        },
        "baseline": {
            "generated_token_ids": baseline_ids.tolist(),
            "response": baseline_response,
        },
        "steered": {
            "generated_token_ids": steered_ids.tolist(),
            "response": steered_response,
        },
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved results to {args.output}")

    print("\nBaseline response:\n")
    print(baseline_response)
    print("\nSteered response:\n")
    print(steered_response)


if __name__ == "__main__":
    main()
