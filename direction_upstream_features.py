"""
Find SAE features causally upstream of an arbitrary residual-stream direction.

This mirrors the LessWrong "Finding Features Causally Upstream of Refusal"
recipe, but replaces the refusal direction with a user-supplied direction.

Method:
1. Load openai/gpt-oss-20b.
2. Load the layer-11 resid_post SAE from
   andyrdt/saes-gpt-oss-20b/resid_post_layer_11/trainer_0.
3. Define a scalar metric as the projection onto a user-provided direction at a
   target layer and token position.
4. Backprop that scalar metric to layer-11 resid_post activations on prompt
   tokens.
5. Rank SAE latents by summed relative gradient:
      sum_t d_i . grad_t
   where d_i is the SAE decoder vector for latent i.
6. Optionally steer generation on prompt tokens only using either the raw
   direction or one SAE latent.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "openai/gpt-oss-20b"
SAE_REPO_ID = "andyrdt/saes-gpt-oss-20b"
SAE_SUBDIR = "resid_post_layer_11/trainer_0"
DEFAULT_OUTPUT_PATH = "direction_upstream_features.json"


@dataclass
class BatchTopKSAE:
    encoder_weight: torch.Tensor
    encoder_bias: torch.Tensor
    decoder_weight: torch.Tensor
    b_dec: torch.Tensor
    threshold: torch.Tensor
    k: int

    @property
    def d_model(self) -> int:
        return self.decoder_weight.shape[0]

    @property
    def d_sae(self) -> int:
        return self.decoder_weight.shape[1]

    @property
    def decoder(self) -> torch.Tensor:
        return self.decoder_weight.transpose(0, 1)

    def to(self, device: torch.device, dtype: torch.dtype | None = None) -> "BatchTopKSAE":
        target_dtype = dtype or self.encoder_weight.dtype
        return BatchTopKSAE(
            encoder_weight=self.encoder_weight.to(device=device, dtype=target_dtype),
            encoder_bias=self.encoder_bias.to(device=device, dtype=target_dtype),
            decoder_weight=self.decoder_weight.to(device=device, dtype=target_dtype),
            b_dec=self.b_dec.to(device=device, dtype=target_dtype),
            threshold=self.threshold.to(device=device, dtype=target_dtype),
            k=self.k,
        )

    def encode_pre_acts(self, resid: torch.Tensor) -> torch.Tensor:
        centered = resid - self.b_dec
        return F.linear(centered, self.encoder_weight, self.encoder_bias)

    def encode(self, resid: torch.Tensor) -> torch.Tensor:
        pre_acts = self.encode_pre_acts(resid)
        acts = F.relu(pre_acts - self.threshold)
        if self.k <= 0 or self.k >= acts.shape[-1]:
            return acts

        topk_vals, topk_idx = torch.topk(acts, k=self.k, dim=-1)
        sparse = torch.zeros_like(acts)
        sparse.scatter_(-1, topk_idx, topk_vals)
        return sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="User prompt to analyze.")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System prompt used for both analysis and generation.",
    )
    parser.add_argument(
        "--direction",
        help="Comma-separated residual direction values. Mutually exclusive with --direction-file.",
    )
    parser.add_argument(
        "--direction-file",
        help="Path to a .pt/.pth/.bin/.npy file containing a 1D residual direction.",
    )
    parser.add_argument(
        "--direction-scale",
        type=float,
        default=1.0,
        help="Multiply the supplied direction by this scalar before analysis.",
    )
    parser.add_argument(
        "--normalize-direction",
        action="store_true",
        help="Normalize the supplied direction to unit norm before applying --direction-scale.",
    )
    parser.add_argument(
        "--sae-layer",
        type=int,
        default=11,
        help="Upstream layer whose resid_post SAE is analyzed. This checkpoint is layer 11.",
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        required=True,
        help="Downstream resid_post layer where the metric projection is measured.",
    )
    parser.add_argument(
        "--target-token",
        default="last",
        help="Prompt token index for the metric. Use 'last' or a zero-based integer.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top and bottom SAE latents to save.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Generation length for optional steering demos.",
    )
    parser.add_argument(
        "--steer-latent",
        type=int,
        help="Optional SAE latent id to steer with on prompt tokens only.",
    )
    parser.add_argument(
        "--steer-direction",
        action="store_true",
        help="If set, also run prompt-only steering with the raw supplied direction.",
    )
    parser.add_argument(
        "--steer-scale",
        type=float,
        default=1.0,
        help="Steering coefficient for either --steer-latent or --steer-direction.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="JSON output path.",
    )

    args = parser.parse_args()
    if bool(args.direction) == bool(args.direction_file):
        parser.error("Specify exactly one of --direction or --direction-file.")
    if args.sae_layer != 11:
        parser.error("This script currently supports the provided layer-11 SAE only.")
    if args.target_layer < args.sae_layer:
        parser.error("--target-layer must be >= --sae-layer so the gradient can flow downstream.")
    return args


def load_direction(args: argparse.Namespace, expected_dim: int, device: torch.device) -> torch.Tensor:
    if args.direction:
        values = [float(x.strip()) for x in args.direction.split(",") if x.strip()]
        direction = torch.tensor(values, dtype=torch.float32)
    else:
        path = Path(args.direction_file)
        suffix = path.suffix.lower()
        if suffix == ".npy":
            import numpy as np

            direction = torch.from_numpy(np.load(path))
        else:
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                if "direction" in obj:
                    obj = obj["direction"]
                elif "tensor" in obj:
                    obj = obj["tensor"]
                else:
                    raise ValueError(
                        f"Direction file {path} is a dict without a 'direction' or 'tensor' key."
                    )
            direction = torch.as_tensor(obj)

    direction = direction.flatten().float()
    if direction.numel() != expected_dim:
        raise ValueError(
            f"Direction has dim {direction.numel()}, but model residual dim is {expected_dim}."
        )
    if args.normalize_direction:
        direction = direction / direction.norm().clamp_min(1e-12)
    direction = direction * args.direction_scale
    return direction.to(device)


def load_sae(device: torch.device, dtype: torch.dtype) -> tuple[BatchTopKSAE, dict[str, Any]]:
    cfg_path = hf_hub_download(repo_id=SAE_REPO_ID, filename=f"{SAE_SUBDIR}/config.json")
    ckpt_path = hf_hub_download(repo_id=SAE_REPO_ID, filename=f"{SAE_SUBDIR}/ae.pt")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    state = torch.load(ckpt_path, map_location="cpu")

    sae = BatchTopKSAE(
        encoder_weight=state["encoder.weight"],
        encoder_bias=state["encoder.bias"],
        decoder_weight=state["decoder.weight"],
        b_dec=state["b_dec"],
        threshold=state["threshold"],
        k=int(state["k"].item() if hasattr(state["k"], "item") else state["k"]),
    ).to(device=device, dtype=dtype)
    return sae, config


def resolve_core_model(model: AutoModelForCausalLM) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise ValueError("Unsupported model structure: expected model.model.layers")


def resolve_prompt_token_index(token_index: str, prompt_len: int) -> int:
    if token_index == "last":
        return prompt_len - 1
    idx = int(token_index)
    if idx < 0:
        idx += prompt_len
    if idx < 0 or idx >= prompt_len:
        raise IndexError(f"Prompt token index {idx} out of range for prompt length {prompt_len}.")
    return idx


def decode_prompt_tokens(tokenizer: AutoTokenizer, input_ids: torch.Tensor) -> list[str]:
    return [tokenizer.decode([tok]) for tok in input_ids.tolist()]


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    prompt_only_hook=None,
) -> str:
    hook_handle = None
    if prompt_only_hook is not None:
        core = resolve_core_model(model)
        hook_handle = core.layers[prompt_only_hook.layer_idx].register_forward_hook(prompt_only_hook)

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    new_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


class PromptOnlySteeringHook:
    def __init__(self, layer_idx: int, vector: torch.Tensor, coeff: float, prompt_len: int):
        self.layer_idx = layer_idx
        self.vector = vector
        self.coeff = coeff
        self.prompt_len = prompt_len

    def __call__(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        seq_len = hidden.shape[1]
        steer_len = min(seq_len, self.prompt_len)
        hidden[:, :steer_len, :] += self.coeff * self.vector.view(1, 1, -1)
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden


def main() -> None:
    args = parse_args()

    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    core = resolve_core_model(model)
    device = model.device
    dtype = next(model.parameters()).dtype
    print(f"Model loaded on: {device}")

    print(f"Loading SAE: {SAE_REPO_ID}/{SAE_SUBDIR}")
    sae, sae_config = load_sae(device=device, dtype=dtype)

    direction = load_direction(args, expected_dim=sae.d_model, device=device).to(dtype=dtype)

    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[-1]
    target_token_idx = resolve_prompt_token_index(args.target_token, prompt_len)

    upstream_resid: torch.Tensor | None = None
    downstream_metric: torch.Tensor | None = None

    def upstream_hook(module, hook_args, output):
        nonlocal upstream_resid
        hidden = output[0] if isinstance(output, tuple) else output
        upstream_resid = hidden

    def target_hook(module, hook_args, output):
        nonlocal downstream_metric
        hidden = output[0] if isinstance(output, tuple) else output
        downstream_metric = hidden[:, target_token_idx, :].mul(direction).sum()

    upstream_handle = core.layers[args.sae_layer].register_forward_hook(upstream_hook)
    target_handle = core.layers[args.target_layer].register_forward_hook(target_hook)

    try:
        outputs = model(**inputs)
        del outputs
    finally:
        upstream_handle.remove()
        target_handle.remove()

    if upstream_resid is None or downstream_metric is None:
        raise RuntimeError("Failed to capture upstream residuals or downstream metric.")

    grad = torch.autograd.grad(downstream_metric, upstream_resid, retain_graph=False)[0][0]
    upstream_prompt_resid = upstream_resid[0, :prompt_len, :].detach()
    grad_prompt = grad[:prompt_len].detach().float()

    with torch.inference_mode():
        latent_acts = sae.encode(upstream_prompt_resid)

    decoder = sae.decoder.detach().float()
    relative_grad = torch.einsum("fd,td->ft", decoder, grad_prompt)
    relative_grad_sum = relative_grad.sum(dim=1)
    relative_grad_mean = relative_grad.mean(dim=1)
    decoder_norm = decoder.norm(dim=1).clamp_min(1e-12)
    grad_norm = grad_prompt.norm(dim=1).clamp_min(1e-12)
    cosine_per_token = relative_grad / (decoder_norm[:, None] * grad_norm[None, :])
    cosine_sum = cosine_per_token.sum(dim=1)

    attribution = (latent_acts.detach().float().transpose(0, 1) * relative_grad).sum(dim=1)

    top_vals, top_idx = torch.topk(relative_grad_sum, k=min(args.top_k, sae.d_sae))
    bot_vals, bot_idx = torch.topk(-relative_grad_sum, k=min(args.top_k, sae.d_sae))

    prompt_tokens = decode_prompt_tokens(tokenizer, inputs["input_ids"][0, :prompt_len].cpu())

    def build_latent_entry(latent_id: int, score: float) -> dict[str, Any]:
        token_rel_grad = relative_grad[latent_id]
        token_attrs = latent_acts[:, latent_id].detach().float() * token_rel_grad
        top_token_vals, top_token_idx = torch.topk(
            token_rel_grad, k=min(5, token_rel_grad.numel())
        )
        top_attr_vals, top_attr_idx = torch.topk(token_attrs, k=min(5, token_attrs.numel()))
        return {
            "latent_id": int(latent_id),
            "relative_gradient_sum": float(score),
            "relative_gradient_mean": float(relative_grad_mean[latent_id].item()),
            "cosine_sum": float(cosine_sum[latent_id].item()),
            "attribution_score": float(attribution[latent_id].item()),
            "max_activation_on_prompt": float(latent_acts[:, latent_id].max().item()),
            "mean_activation_on_prompt": float(latent_acts[:, latent_id].mean().item()),
            "top_relative_gradient_tokens": [
                {
                    "token_index": int(idx),
                    "token_text": prompt_tokens[idx],
                    "relative_gradient": float(val),
                }
                for val, idx in zip(top_token_vals.tolist(), top_token_idx.tolist())
            ],
            "top_attribution_tokens": [
                {
                    "token_index": int(idx),
                    "token_text": prompt_tokens[idx],
                    "attribution": float(val),
                }
                for val, idx in zip(top_attr_vals.tolist(), top_attr_idx.tolist())
            ],
        }

    top_latents = [build_latent_entry(i, v) for i, v in zip(top_idx.tolist(), top_vals.tolist())]
    bottom_latents = [build_latent_entry(i, -v) for i, v in zip(bot_idx.tolist(), bot_vals.tolist())]

    baseline_response = generate_response(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        max_new_tokens=args.max_new_tokens,
    )

    steering_results: dict[str, Any] = {
        "baseline_response": baseline_response,
    }

    if args.steer_direction:
        hook = PromptOnlySteeringHook(
            layer_idx=args.sae_layer,
            vector=direction,
            coeff=args.steer_scale,
            prompt_len=prompt_len,
        )
        steering_results["direction_steering"] = {
            "scale": args.steer_scale,
            "response": generate_response(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                max_new_tokens=args.max_new_tokens,
                prompt_only_hook=hook,
            ),
        }

    if args.steer_latent is not None:
        latent_decoder = decoder[args.steer_latent].to(device=device, dtype=dtype)
        hook = PromptOnlySteeringHook(
            layer_idx=args.sae_layer,
            vector=latent_decoder,
            coeff=args.steer_scale,
            prompt_len=prompt_len,
        )
        steering_results["latent_steering"] = {
            "latent_id": args.steer_latent,
            "scale": args.steer_scale,
            "response": generate_response(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                max_new_tokens=args.max_new_tokens,
                prompt_only_hook=hook,
            ),
        }

    results = {
        "model_id": MODEL_ID,
        "sae_repo_id": SAE_REPO_ID,
        "sae_subdir": SAE_SUBDIR,
        "sae_config": sae_config,
        "prompt": args.prompt,
        "system_prompt": args.system_prompt,
        "prompt_length": prompt_len,
        "prompt_tokens": [
            {"token_index": i, "token_text": tok} for i, tok in enumerate(prompt_tokens)
        ],
        "direction_info": {
            "source": "inline" if args.direction else str(Path(args.direction_file).resolve()),
            "norm": float(direction.float().norm().item()),
            "normalized": bool(args.normalize_direction),
            "direction_scale": args.direction_scale,
        },
        "metric": {
            "type": "projection",
            "sae_layer": args.sae_layer,
            "target_layer": args.target_layer,
            "target_token_index": target_token_idx,
            "target_token_text": prompt_tokens[target_token_idx],
            "projection_value": float(downstream_metric.detach().float().item()),
        },
        "top_latents_by_relative_gradient_sum": top_latents,
        "bottom_latents_by_relative_gradient_sum": bottom_latents,
        "steering": steering_results,
    }

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {output_path.resolve()}")
    print("Top latents by summed relative gradient:")
    for entry in top_latents[: min(10, len(top_latents))]:
        print(
            f"  latent {entry['latent_id']:>6} | RG sum {entry['relative_gradient_sum']:+.5f} "
            f"| attr {entry['attribution_score']:+.5f}"
        )


if __name__ == "__main__":
    main()
