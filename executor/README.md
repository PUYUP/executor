# Executor

Curiosift Indexer Executor component.

## Setup

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Usage

```bash
python -m executor.main
```

## Development

```bash
# Run tests
pytest

# Lint
ruff check .

# Type check
mypy executor
```
