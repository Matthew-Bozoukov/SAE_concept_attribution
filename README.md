## Setup

This repo is configured for [uv](https://docs.astral.sh/uv/).

Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create or update the local environment from `pyproject.toml` and `uv.lock`:

```bash
uv sync
```

Run scripts through the uv environment:

```bash
uv run python llama33_feature_response_cosine.py --help
```
