"""
Replicate the LessWrong-style algorithmic CoT clustering pipeline on a single
GSM8K question using gpt-oss-20b.

Pipeline:
1) Sample many chain-of-thought rollouts for one fixed prompt.
2) Ask the model to infer a compact set of reasoning strategies + unique keywords
   from a subset of rollouts.
3) Label rollout steps by keyword matching.
4) Build an algorithm graph (strategy transition graph).
5) Save all artifacts to JSON.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from urllib import error, request
from dataclasses import dataclass
from typing import Any

from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "openai/gpt-oss-20b"
QUESTION = (
    "James decides to run 3 sprints 3 times a week. "
    "He runs 60 meters each sprint. "
    "How many total meters does he run a week?"
)
SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, showing your reasoning clearly. "
    "End your response with 'The answer is: <number>'."
)

NUM_ROLLOUTS = 100
SUBSET_FOR_STRATEGY_DISCOVERY = 10
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.8
TOP_P = 0.95

MAX_STRATEGIES = 8
MAX_KEYWORDS_PER_STRATEGY = 8

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

OUTPUT_JSON_PATH = "gpt_oss_20b_algorithmic_clustering_gsm8k.json"


@dataclass
class Strategy:
    name: str
    description: str
    keywords: list[str]


def split_into_steps(text: str) -> list[str]:
    """Lightweight step chunking: newline-aware, then sentence-ish splitting."""
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line)
        for part in parts:
            part = part.strip(" -\t")
            if part:
                chunks.append(part)
    return chunks


def generate_rollout(
    llm: LLM,
    tokenizer: AutoTokenizer,
    question: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    outputs = llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text.strip()


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse JSON robustly even if the model wraps it with prose/code fences."""
    raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, flags=re.DOTALL)
    if fenced:
        data = json.loads(fenced.group(1))
        if isinstance(data, dict):
            return data

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        data = json.loads(raw_text[start : end + 1])
        if isinstance(data, dict):
            return data

    raise ValueError("Could not parse JSON object from model output")


def discover_strategies_with_openrouter(
    subset_rollouts: list[str],
    max_strategies: int,
    max_keywords_per_strategy: int,
) -> list[Strategy]:
    api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set {OPENROUTER_API_KEY_ENV} to use OpenRouter strategy discovery")

    discovery_prompt = (
        "You are analyzing many chain-of-thought rollouts for one math question.\n"
        "Infer recurring high-level reasoning strategies and assign distinct keyword cues.\n\n"
        "Return STRICT JSON only with this schema:\n"
        "{\n"
        '  "strategies": [\n'
        "    {\n"
        '      "name": "short strategy name",\n'
        '      "description": "one sentence",\n'
        '      "keywords": ["keyword1", "keyword2"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Constraints:\n"
        f"- At most {max_strategies} strategies.\n"
        f"- At most {max_keywords_per_strategy} lowercase keywords per strategy.\n"
        "- Keywords should be specific lexical cues likely to appear in that strategy.\n"
        "- Avoid overlapping keywords across strategies when possible.\n"
        "- Output JSON only, no prose.\n\n"
        "Rollouts:\n"
    )

    for i, rollout in enumerate(subset_rollouts, start=1):
        trimmed = rollout.strip()[:1800]
        discovery_prompt += f"\n[ROLLOUT {i}]\n{trimmed}\n"

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You extract strategy taxonomies from model rollouts."},
            {"role": "user", "content": discovery_prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    req = request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120) as resp:
            response_json = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter request failed with HTTP {exc.code}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    try:
        raw = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {response_json}") from exc

    if isinstance(raw, list):
        raw = "".join(part.get("text", "") for part in raw if isinstance(part, dict))

    parsed = _extract_json_object(str(raw))

    strategies: list[Strategy] = []
    for s in parsed.get("strategies", []):
        name = str(s.get("name", "")).strip()
        description = str(s.get("description", "")).strip()
        keywords = [str(k).strip().lower() for k in s.get("keywords", []) if str(k).strip()]
        dedup_keywords = list(dict.fromkeys(keywords))[:max_keywords_per_strategy]
        if name and dedup_keywords:
            strategies.append(Strategy(name=name, description=description, keywords=dedup_keywords))

    return strategies[:max_strategies]


def label_step(step_text: str, strategies: list[Strategy]) -> str:
    text = step_text.lower()
    best_name = "unlabeled"
    best_score = 0

    for strat in strategies:
        score = 0
        for kw in strat.keywords:
            if kw and kw in text:
                score += 1
        if score > best_score:
            best_score = score
            best_name = strat.name

    return best_name


def main() -> None:
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"Loading vLLM engine: {MODEL_ID}")
    llm = LLM(model=MODEL_ID)
    print("vLLM engine loaded.\n")

    print(f"Sampling {NUM_ROLLOUTS} rollouts...")
    rollouts: list[str] = []
    for _ in tqdm(range(NUM_ROLLOUTS), desc="Rollouts", unit="rollout"):
        rollout = generate_rollout(
            llm=llm,
            tokenizer=tokenizer,
            question=QUESTION,
            system_prompt=SYSTEM_PROMPT,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        rollouts.append(rollout)

    subset = rollouts[: min(SUBSET_FOR_STRATEGY_DISCOVERY, len(rollouts))]
    print(f"Discovering strategies from {len(subset)} rollouts...")
    strategies = discover_strategies_with_openrouter(
        subset_rollouts=subset,
        max_strategies=MAX_STRATEGIES,
        max_keywords_per_strategy=MAX_KEYWORDS_PER_STRATEGY,
    )
    if not strategies:
        raise RuntimeError("No strategies discovered; cannot continue clustering")

    print("Discovered strategies:")
    for s in strategies:
        print(f"- {s.name}: {', '.join(s.keywords)}")

    labeled_rollouts: list[dict[str, Any]] = []
    node_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()

    for i, rollout in enumerate(rollouts, start=1):
        steps = split_into_steps(rollout)
        labels = [label_step(step, strategies) for step in steps]

        for label in labels:
            node_counts[label] += 1

        for a, b in zip(labels, labels[1:]):
            edge_counts[(a, b)] += 1

        labeled_rollouts.append(
            {
                "rollout_id": i,
                "text": rollout,
                "steps": steps,
                "labels": labels,
            }
        )

    nodes = []
    for s in strategies:
        nodes.append(
            {
                "name": s.name,
                "description": s.description,
                "keywords": s.keywords,
                "step_count": int(node_counts.get(s.name, 0)),
            }
        )
    nodes.append({"name": "unlabeled", "description": "No keyword match", "keywords": [], "step_count": int(node_counts.get("unlabeled", 0))})

    edges = [
        {
            "source": src,
            "target": dst,
            "count": int(count),
        }
        for (src, dst), count in edge_counts.items()
    ]
    edges.sort(key=lambda x: x["count"], reverse=True)

    payload = {
        "model_id": MODEL_ID,
        "question": QUESTION,
        "system_prompt": SYSTEM_PROMPT,
        "method": "algorithmic_step_clustering_keyword_matching",
        "config": {
            "num_rollouts": NUM_ROLLOUTS,
            "subset_for_strategy_discovery": SUBSET_FOR_STRATEGY_DISCOVERY,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_strategies": MAX_STRATEGIES,
            "max_keywords_per_strategy": MAX_KEYWORDS_PER_STRATEGY,
            "strategy_discovery_model": OPENROUTER_MODEL,
        },
        "strategies": [
            {
                "name": s.name,
                "description": s.description,
                "keywords": s.keywords,
            }
            for s in strategies
        ],
        "graph": {
            "nodes": nodes,
            "edges": edges,
        },
        "rollouts": labeled_rollouts,
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved clustering output to: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
