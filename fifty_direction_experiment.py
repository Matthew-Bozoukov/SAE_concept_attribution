"""
Estimate a residual-stream "50 direction" from prompt states and test whether
steering along it increases the probability that the model answers 50.

Method:
1. Build addition prompts, some whose sum is 50 and some whose sum is not 50.
2. Capture the target-layer residual at the last prompt token (the token right
   before generation, e.g. after the trailing "=").
3. Define the "50 direction" as mean(resid | prompt sum == 50) -
   mean(resid | prompt sum != 50).
4. Steer prompt tokens only along that direction and compare the answer-50 rate
   before vs after steering on held-out addition prompts.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_ID = "openai/gpt-oss-20b"
DEFAULT_QUESTION = "Given x+y=10, find minimum of x^2+y^2."
DEFAULT_SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem carefully and end your response with "
    "'The answer is: <number>'."
)
DEFAULT_OUTPUT = "fifty_direction_experiment.json"
DEFAULT_TEST_SYSTEM_PROMPT = "You are a helpful assistant. Answer with just the number."
DEFAULT_DIRECTION_SYSTEM_PROMPT = "You are a helpful assistant. Complete the arithmetic expression."


@dataclass
class Sample:
    seed: int
    response: str
    parsed_answer: str | None
    final_token_resid: torch.Tensor


class PromptOnlySteeringHook:
    def __init__(self, layer_idx: int, vector: torch.Tensor, coeff: float, prompt_len: int):
        self.layer_idx = layer_idx
        self.vector = vector
        self.coeff = coeff
        self.prompt_len = prompt_len

    def __call__(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        steer_len = min(hidden.shape[1], self.prompt_len)
        hidden[:, :steer_len, :] += self.coeff * self.vector.view(1, 1, -1)
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question', default=DEFAULT_QUESTION)
    parser.add_argument('--system-prompt', default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument('--test-system-prompt', default=DEFAULT_TEST_SYSTEM_PROMPT)
    parser.add_argument('--direction-system-prompt', default=DEFAULT_DIRECTION_SYSTEM_PROMPT)
    parser.add_argument('--target-answer', default='50')
    parser.add_argument('--layer', type=int, default=20)
    parser.add_argument('--layers', default='', help='Comma-separated layer sweep, e.g. 10,20,30. If unset, uses --layer.')
    parser.add_argument('--num-samples', type=int, default=40)
    parser.add_argument('--num-test-samples', type=int, default=40)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--max-new-tokens', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for generation and residual capture forward passes')
    parser.add_argument('--steer-scales', default='4.0', help='Comma-separated steering scales to test, e.g. 0,1,2,4,8')
    parser.add_argument('--normalize-direction', action='store_true')
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--test-mode', choices=['same_prompt', 'random_addition'], default='random_addition')
    parser.add_argument('--addition-min', type=int, default=0)
    parser.add_argument('--addition-max', type=int, default=99)
    parser.add_argument('--direction-min', type=int, default=0)
    parser.add_argument('--direction-max', type=int, default=49)
    parser.add_argument('--direction-seed', type=int, default=12345)
    return parser.parse_args()


def parse_steer_scales(raw: str) -> list[float]:
    scales = [float(x.strip()) for x in raw.split(',') if x.strip()]
    if not scales:
        raise ValueError('At least one steering scale must be provided.')
    return scales


def parse_layers(raw: str, fallback_layer: int) -> list[int]:
    if not raw.strip():
        return [fallback_layer]
    layers = [int(x.strip()) for x in raw.split(',') if x.strip()]
    if not layers:
        raise ValueError('At least one layer must be provided.')
    return layers


def resolve_core_model(model: AutoModelForCausalLM) -> Any:
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model
    raise ValueError('Unsupported model structure: expected model.model.layers')


def build_inputs(tokenizer: AutoTokenizer, system_prompt: str, question: str) -> dict[str, torch.Tensor]:
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(prompt, return_tensors='pt')


def extract_numeric_answer(text: str) -> str | None:
    markers = [
        r'The answer is:\s*([-+]?\d+(?:\.\d+)?)',
        r'answer\s*[:=]\s*([-+]?\d+(?:\.\d+)?)',
    ]
    for pattern in markers:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1]
    nums = re.findall(r'[-+]?\d+(?:\.\d+)?', text)
    return nums[-1] if nums else None


def build_random_addition_questions(num_questions: int, low: int, high: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    questions: list[str] = []
    for _ in range(num_questions):
        a = rng.randint(low, high)
        b = rng.randint(low, high)
        questions.append(f"What is {a}+{b}?")
    return questions


def build_direction_questions(num_questions: int, low: int, high: int, seed: int = 0, target_sum: int = 50) -> tuple[list[str], list[str]]:
    if low > high:
        raise ValueError('direction-min must be <= direction-max')
    if target_sum - high > target_sum - low:
        raise ValueError('Invalid direction range')
    rng = random.Random(seed)
    positives: list[str] = []
    negatives: list[str] = []
    midpoint = num_questions // 2

    while len(positives) < midpoint:
        a = rng.randint(low, high)
        b = target_sum - a
        if low <= b <= high:
            positives.append(f"{a}+{b}=")

    while len(negatives) < num_questions - midpoint:
        a = rng.randint(low, high)
        b = rng.randint(low, high)
        if a + b != target_sum:
            negatives.append(f"{a}+{b}=")

    questions = positives + negatives
    labels = [str(target_sum)] * len(positives) + ['not_50'] * len(negatives)
    combined = list(zip(questions, labels))
    rng.shuffle(combined)
    shuffled_questions, shuffled_labels = zip(*combined)
    return list(shuffled_questions), list(shuffled_labels)


def build_batched_inputs(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    questions: list[str],
) -> tuple[dict[str, torch.Tensor], list[int]]:
    tokenized = [build_inputs(tokenizer, system_prompt, question) for question in questions]
    max_prompt_len = max(item['input_ids'].shape[1] for item in tokenized)
    pad_id = tokenizer.eos_token_id
    batch_size = len(tokenized)
    batch_input_ids = torch.full((batch_size, max_prompt_len), pad_id, dtype=tokenized[0]['input_ids'].dtype)
    batch_attention_mask = torch.zeros((batch_size, max_prompt_len), dtype=torch.long)
    prompt_lens: list[int] = []
    for row, item in enumerate(tokenized):
        prompt_len = item['input_ids'].shape[1]
        prompt_lens.append(prompt_len)
        batch_input_ids[row, :prompt_len] = item['input_ids'][0]
        batch_attention_mask[row, :prompt_len] = 1
    return {'input_ids': batch_input_ids, 'attention_mask': batch_attention_mask}, prompt_lens


def collect_prompt_state_samples_for_questions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    questions: list[str],
    labels: list[str],
    system_prompt: str,
    layer: int,
    seed_offset: int = 0,
    desc: str = 'Prompt-state sampling',
    batch_size: int = 8,
) -> list[Sample]:
    if len(questions) != len(labels):
        raise ValueError('questions and labels must have the same length')

    samples: list[Sample] = []
    num_batches = (len(questions) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc=desc, unit='batch'):
        start = batch_idx * batch_size
        batch_questions = questions[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]
        inputs, _ = build_batched_inputs(tokenizer, system_prompt, batch_questions)
        resid_batch = capture_final_token_resid_batch(
            model,
            inputs['input_ids'],
            inputs['attention_mask'],
            layer,
        )
        for offset, (question, label, resid) in enumerate(zip(batch_questions, batch_labels, resid_batch.unbind(dim=0))):
            samples.append(
                Sample(
                    seed=seed_offset + start + offset,
                    response=question,
                    parsed_answer=label,
                    final_token_resid=resid,
                )
            )
    return samples


def collect_samples_for_questions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    questions: list[str],
    system_prompt: str,
    layer: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed_offset: int = 0,
    prompt_only_hook=None,
    desc: str = 'Sampling',
    batch_size: int = 8,
) -> list[Sample]:
    samples: list[Sample] = []
    num_batches = (len(questions) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f'{desc} generation', unit='batch'):
        start = batch_idx * batch_size
        batch_questions = questions[start:start + batch_size]
        current_batch_size = len(batch_questions)
        inputs, prompt_lens = build_batched_inputs(tokenizer, system_prompt, batch_questions)
        pad_id = tokenizer.eos_token_id
        batch_prompt_hook = prompt_only_hook
        if prompt_only_hook is not None:
            batch_prompt_hook = PromptOnlySteeringHook(
                layer_idx=prompt_only_hook.layer_idx,
                vector=prompt_only_hook.vector,
                coeff=prompt_only_hook.coeff,
                prompt_len=max(prompt_lens),
            )

        responses, full_ids_batch = generate_responses_batch(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed_offset + batch_idx,
            prompt_only_hook=batch_prompt_hook,
        )

        max_len = max(ids.shape[0] for ids in full_ids_batch)
        full_batch_ids = torch.full((current_batch_size, max_len), pad_id, dtype=full_ids_batch[0].dtype)
        full_attention_mask = torch.zeros((current_batch_size, max_len), dtype=torch.long)
        for row, ids in enumerate(full_ids_batch):
            seq_len = ids.shape[0]
            full_batch_ids[row, :seq_len] = ids
            full_attention_mask[row, :seq_len] = 1
        final_resids = capture_final_token_resid_batch(model, full_batch_ids, full_attention_mask, layer)

        for offset, (response, resid) in enumerate(zip(responses, final_resids.unbind(dim=0))):
            seed = seed_offset + start + offset
            parsed = extract_numeric_answer(response)
            samples.append(Sample(seed=seed, response=response, parsed_answer=parsed, final_token_resid=resid))

    return samples


def generate_responses_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    prompt_only_hook=None,
) -> tuple[list[str], list[torch.Tensor]]:
    core = resolve_core_model(model)
    device = model.device
    hook_handle = None
    if prompt_only_hook is not None:
        hook_handle = core.layers[prompt_only_hook.layer_idx].register_forward_hook(prompt_only_hook)

    batched_inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    try:
        if hasattr(torch, 'manual_seed'):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        with torch.inference_mode():
            output_ids = model.generate(
                **batched_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    prompt_len = batched_inputs['input_ids'].shape[-1]
    responses = [
        tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
        for row in output_ids
    ]
    full_ids = [row.detach().cpu() for row in output_ids]
    return responses, full_ids


def capture_final_token_resid_batch(
    model: AutoModelForCausalLM,
    full_ids_batch: torch.Tensor,
    attention_mask: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    core = resolve_core_model(model)
    captured: torch.Tensor | None = None

    def hook_fn(module, args, output):
        nonlocal captured
        hidden = output[0] if isinstance(output, tuple) else output
        captured = hidden.detach()

    handle = core.layers[layer].register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            model(
                input_ids=full_ids_batch.to(model.device),
                attention_mask=attention_mask.to(model.device),
            )
    finally:
        handle.remove()

    if captured is None:
        raise RuntimeError('Failed to capture final-token residual batch.')

    last_indices = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(captured.shape[0], device=captured.device)
    final_resid = captured[batch_indices, last_indices, :]
    return final_resid.float().cpu()


def collect_samples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: dict[str, torch.Tensor],
    layer: int,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed_offset: int = 0,
    prompt_only_hook=None,
    desc: str = 'Sampling',
    batch_size: int = 8,
) -> list[Sample]:
    samples_meta: list[dict[str, Any]] = []
    generated_ids: list[torch.Tensor] = []

    num_batches = (num_samples + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc=f'{desc} generation', unit='batch'):
        start = batch_idx * batch_size
        current_batch_size = min(batch_size, num_samples - start)
        batch_inputs = {
            key: value.repeat(current_batch_size, 1)
            for key, value in inputs.items()
        }
        responses, full_ids_batch = generate_responses_batch(
            model=model,
            tokenizer=tokenizer,
            inputs=batch_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed_offset + batch_idx,
            prompt_only_hook=prompt_only_hook,
        )
        for offset, (response, full_ids) in enumerate(zip(responses, full_ids_batch)):
            seed = seed_offset + start + offset
            samples_meta.append({
                'seed': seed,
                'response': response,
                'parsed_answer': extract_numeric_answer(response),
            })
            generated_ids.append(full_ids)

    final_resids: list[torch.Tensor] = []
    for start in tqdm(range(0, len(generated_ids), batch_size), desc=f'{desc} activations', unit='batch'):
        batch = generated_ids[start:start + batch_size]
        max_len = max(ids.shape[0] for ids in batch)
        pad_id = tokenizer.eos_token_id
        batch_ids = torch.full((len(batch), max_len), pad_id, dtype=batch[0].dtype)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row, ids in enumerate(batch):
            seq_len = ids.shape[0]
            batch_ids[row, :seq_len] = ids
            attention_mask[row, :seq_len] = 1
        batch_resids = capture_final_token_resid_batch(model, batch_ids, attention_mask, layer)
        final_resids.extend(batch_resids.unbind(dim=0))

    samples: list[Sample] = []
    for meta, resid in zip(samples_meta, final_resids):
        samples.append(
            Sample(
                seed=meta['seed'],
                response=meta['response'],
                parsed_answer=meta['parsed_answer'],
                final_token_resid=resid,
            )
        )
    return samples


def summarize_answers(samples: list[Sample]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for sample in samples:
        counter[sample.parsed_answer or '<none>'] += 1
    return dict(counter)


def main() -> None:
    args = parse_args()

    print(f'Loading model: {MODEL_ID}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    print(f'Model loaded on: {model.device}')

    inputs = build_inputs(tokenizer, args.system_prompt, args.question)
    steer_scales = parse_steer_scales(args.steer_scales)
    layers = parse_layers(args.layers, args.layer)
    test_questions = [args.question]
    if args.test_mode == 'random_addition':
        test_questions = build_random_addition_questions(
            num_questions=args.num_test_samples,
            low=args.addition_min,
            high=args.addition_max,
            seed=12345,
        )

    direction_questions, direction_labels = build_direction_questions(
        num_questions=args.num_samples,
        low=args.direction_min,
        high=args.direction_max,
        seed=args.direction_seed,
        target_sum=int(args.target_answer),
    )

    layer_results = []
    for layer in layers:
        print()
        print(f'===== Layer {layer} =====')
        print(
            f'Collecting {args.num_samples} prompt-state samples to estimate the {args.target_answer} direction '
            'from addition expressions...'
        )
        baseline_samples = collect_prompt_state_samples_for_questions(
            model=model,
            tokenizer=tokenizer,
            questions=direction_questions,
            labels=direction_labels,
            system_prompt=args.direction_system_prompt,
            layer=layer,
            desc=f'Direction estimation prompt states (L{layer})',
            batch_size=args.batch_size,
        )

        positive_label = args.target_answer
        negatives = [s.final_token_resid for s in baseline_samples if s.parsed_answer != positive_label]
        positives = [s.final_token_resid for s in baseline_samples if s.parsed_answer == positive_label]

        if not positives:
            raise RuntimeError(f'No prompt-state positives for target answer {args.target_answer!r} at layer {layer}.')
        if not negatives:
            raise RuntimeError(f'No prompt-state negatives were created for contrastive direction estimation at layer {layer}.')

        pos_mean = torch.stack(positives).mean(dim=0)
        neg_mean = torch.stack(negatives).mean(dim=0)
        direction = pos_mean - neg_mean
        raw_norm = float(direction.norm().item())
        if args.normalize_direction:
            direction = direction / direction.norm().clamp_min(1e-12)

        device = model.device
        dtype = next(model.parameters()).dtype
        direction_vector = direction.to(device=device, dtype=dtype)

        print(f'Testing baseline on {args.num_test_samples} samples using {args.test_mode}...')
        if args.test_mode == 'same_prompt':
            baseline_test_samples = collect_samples(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                layer=layer,
                num_samples=args.num_test_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed_offset=10_000,
                desc=f'Baseline test (L{layer})',
                batch_size=args.batch_size,
            )
        else:
            baseline_test_samples = collect_samples_for_questions(
                model=model,
                tokenizer=tokenizer,
                questions=test_questions,
                system_prompt=args.test_system_prompt,
                layer=layer,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed_offset=10_000,
                desc=f'Baseline test (L{layer})',
                batch_size=args.batch_size,
            )
        baseline_hits = sum(s.parsed_answer == args.target_answer for s in baseline_test_samples)

        steering_results = []
        for idx, scale in enumerate(steer_scales, start=1):
            print(f'Testing steered samples at scale {scale} on {args.num_test_samples} additional samples...')
            prompt_len = inputs['input_ids'].shape[-1]
            steering_hook = PromptOnlySteeringHook(
                layer_idx=layer,
                vector=direction_vector,
                coeff=scale,
                prompt_len=prompt_len,
            )
            if args.test_mode == 'same_prompt':
                steered_test_samples = collect_samples(
                    model=model,
                    tokenizer=tokenizer,
                    inputs=inputs,
                    layer=layer,
                    num_samples=args.num_test_samples,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed_offset=20_000 + idx * 10_000,
                    prompt_only_hook=steering_hook,
                    desc=f'Steered test L{layer} @ {scale}',
                    batch_size=args.batch_size,
                )
            else:
                steered_test_samples = collect_samples_for_questions(
                    model=model,
                    tokenizer=tokenizer,
                    questions=test_questions,
                    system_prompt=args.test_system_prompt,
                    layer=layer,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed_offset=20_000 + idx * 10_000,
                    prompt_only_hook=steering_hook,
                    desc=f'Steered test L{layer} @ {scale}',
                    batch_size=args.batch_size,
                )
            steered_hits = sum(s.parsed_answer == args.target_answer for s in steered_test_samples)
            steering_results.append({
                'scale': scale,
                'answer_counts': summarize_answers(steered_test_samples),
                'target_answer_rate': steered_hits / len(steered_test_samples),
                'samples': [
                    {'seed': s.seed, 'parsed_answer': s.parsed_answer, 'response': s.response}
                    for s in steered_test_samples
                ],
            })

        layer_results.append({
            'layer': layer,
            'direction': {
                'norm_before_optional_normalization': raw_norm,
                'normalized': bool(args.normalize_direction),
                'steer_scales': steer_scales,
                'num_positive_samples': len(positives),
                'num_negative_samples': len(negatives),
            },
            'baseline_direction_estimation_answer_counts': summarize_answers(baseline_samples),
            'baseline_test': {
                'answer_counts': summarize_answers(baseline_test_samples),
                'target_answer_rate': baseline_hits / len(baseline_test_samples),
                'samples': [
                    {'seed': s.seed, 'parsed_answer': s.parsed_answer, 'response': s.response}
                    for s in baseline_test_samples
                ],
            },
            'steered_tests': steering_results,
        })

        print(
            f'Baseline target-answer rate at layer {layer}: {baseline_hits}/{len(baseline_test_samples)} '
            f'({baseline_hits / len(baseline_test_samples):.1%})'
        )
        for item in steering_results:
            rate = item['target_answer_rate']
            hits = sum(1 for sample in item['samples'] if sample['parsed_answer'] == args.target_answer)
            print(f'Steered target-answer rate at layer {layer} @ {item["scale"]}: {hits}/{len(item["samples"])} ({rate:.1%})')

    results = {
        'model_id': MODEL_ID,
        'question': args.question,
        'system_prompt': args.system_prompt,
        'direction_system_prompt': args.direction_system_prompt,
        'target_answer': args.target_answer,
        'layers': layers,
        'num_samples_for_direction': args.num_samples,
        'num_test_samples': args.num_test_samples,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'max_new_tokens': args.max_new_tokens,
        'batch_size': args.batch_size,
        'test_mode': args.test_mode,
        'test_system_prompt': args.test_system_prompt,
        'addition_range': [args.addition_min, args.addition_max],
        'direction_range': [args.direction_min, args.direction_max],
        'test_questions_preview': test_questions[:10],
        'direction_questions_preview': direction_questions[:10],
        'layer_results': layer_results,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2))
    print(f'Saved results to {output_path}')


if __name__ == '__main__':
    main()
