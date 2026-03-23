import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-3-4b-it"
NUM_EXAMPLES = 5
MAX_NEW_TOKENS = 512

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


def generate_answer(question: str, tokenizer, model) -> str:
    messages = [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def extract_gt_answer(answer_str: str) -> str:
    match = re.search(r"####\s*(.+)", answer_str)
    return match.group(1).strip() if match else answer_str.strip()


def extract_model_answer(response: str) -> str:
    match = re.search(r"[Tt]he answer is[:\s]+([\d,\.]+)", response)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"[\d,]+", response)
    return numbers[-1].replace(",", "") if numbers else ""


def main():
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    examples = dataset.select(range(NUM_EXAMPLES))

    tokenizer, model = load_model()

    results = []
    for i, example in enumerate(examples):
        question = example["question"]
        gt_answer = extract_gt_answer(example["answer"])

        print(f"{'='*70}")
        print(f"Question {i+1}: {question}")
        print(f"\nGround truth: {gt_answer}")
        print("\nGenerating response...")

        response = generate_answer(question, tokenizer, model)
        print(f"\nModel response:\n{response}")

        results.append({
            "question": question,
            "ground_truth": gt_answer,
            "model_response": response,
        })

    print(f"\n{'='*70}")
    print("Results summary:")
    print(f"{'Q':>2}  {'Ground Truth':>15}  {'Model Answer':>15}  Correct?")
    print("-" * 50)

    correct = 0
    for i, r in enumerate(results):
        gt = r["ground_truth"].replace(",", "")
        pred = extract_model_answer(r["model_response"])
        is_correct = gt == pred
        if is_correct:
            correct += 1
        print(f"{i+1:>2}  {gt:>15}  {pred:>15}  {'Y' if is_correct else 'N'}")

    print("-" * 50)
    print(f"Accuracy: {correct}/{len(results)} = {correct/len(results):.0%}")


if __name__ == "__main__":
    main()
