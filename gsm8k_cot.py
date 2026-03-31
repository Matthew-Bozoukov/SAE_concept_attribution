import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "openai/gpt-oss-20b"
QUESTION = "Given x+y=10, find minimum of x^2+y^2."
MAX_NEW_TOKENS = 512
PREFILL_SECOND_HALF = True
SECOND_HALF_PREFILL = "Wait, let's try a slightly different approach: "
SENTENCE_END_CHARS = ".!?"
ANALYSIS_MARKER = "analysis"
FINAL_MARKER = "assistantfinal"

COT_SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, showing your reasoning clearly. "
    "Think through each step before giving the final answer. "
    "End your response with 'The answer is: <number>'"
)


def load_model():
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model loaded on: {model.device}\n")
    return tokenizer, model


def build_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_from_prompt(prompt: str, tokenizer, model, max_new_tokens: int) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    return output_ids[0][inputs["input_ids"].shape[-1] :]


def decode_tokens(token_ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def find_sentence_boundary(text: str) -> int:
    midpoint_char = len(text) // 2
    split_char = midpoint_char

    while split_char < len(text) and text[split_char] not in SENTENCE_END_CHARS:
        split_char += 1

    if split_char < len(text):
        split_char += 1
        while split_char < len(text) and text[split_char].isspace():
            split_char += 1
    else:
        split_char = len(text)

    return split_char


def split_analysis_at_sentence_midpoint(baseline_text: str):
    analysis_start = baseline_text.find(ANALYSIS_MARKER)
    final_start = baseline_text.find(FINAL_MARKER)

    if analysis_start == -1 or final_start == -1 or final_start <= analysis_start:
        return None

    analysis_content_start = analysis_start + len(ANALYSIS_MARKER)
    analysis_text = baseline_text[analysis_content_start:final_start]
    split_char = find_sentence_boundary(analysis_text)

    return {
        "analysis_prefix": analysis_text[:split_char],
        "analysis_suffix": analysis_text[split_char:],
        "final_text": baseline_text[final_start:],
    }


def generate_answer(question: str, tokenizer, model) -> str:
    prompt = build_prompt(question, tokenizer)
    new_ids = generate_from_prompt(prompt, tokenizer, model, MAX_NEW_TOKENS)
    return decode_tokens(new_ids, tokenizer)


def generate_answer_with_second_half_prefill(question: str, tokenizer, model):
    prompt = build_prompt(question, tokenizer)
    baseline_ids = generate_from_prompt(prompt, tokenizer, model, MAX_NEW_TOKENS)
    baseline_text = decode_tokens(baseline_ids, tokenizer)

    split_parts = split_analysis_at_sentence_midpoint(baseline_text)
    if split_parts is None:
        raise ValueError(
            "Could not find an 'analysis ... assistantfinal' boundary in the model output."
        )

    analysis_prefix = split_parts["analysis_prefix"]
    analysis_prefill_prompt = prompt + ANALYSIS_MARKER + analysis_prefix + SECOND_HALF_PREFILL

    prefix_text = ANALYSIS_MARKER + analysis_prefix
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0]
    remaining_budget = max(1, MAX_NEW_TOKENS - prefix_ids.shape[-1])

    nudged_suffix_ids = generate_from_prompt(
        analysis_prefill_prompt, tokenizer, model, remaining_budget
    )
    nudged_text = (
        ANALYSIS_MARKER
        + analysis_prefix
        + SECOND_HALF_PREFILL
        + decode_tokens(nudged_suffix_ids, tokenizer)
    )

    return {
        "baseline": baseline_text,
        "analysis_prefix": ANALYSIS_MARKER + analysis_prefix,
        "nudged": nudged_text,
    }


def main():
    tokenizer, model = load_model()
    print(f"Question: {QUESTION}\n")
    print("Generating response...")

    if PREFILL_SECOND_HALF:
        results = generate_answer_with_second_half_prefill(QUESTION, tokenizer, model)
        print("\nBaseline response:\n")
        print(results["baseline"])
        print("\nAnalysis prefix reused for intervention:\n")
        print(results["analysis_prefix"])
        print("\nNudged response with analysis prefill:\n")
        print(results["nudged"])
    else:
        response = generate_answer(QUESTION, tokenizer, model)
        print("\nFull model response:\n")
        print(response)


if __name__ == "__main__":
    main()
