import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "openai/gpt-oss-20b"
QUESTION = "Given x+y=10, find minimum of x^2+y^2."
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


def main():
    tokenizer, model = load_model()
    print(f"Question: {QUESTION}\n")
    print("Generating response...")
    response = generate_answer(QUESTION, tokenizer, model)
    print("\nFull model response:\n")
    print(response)


if __name__ == "__main__":
    main()
