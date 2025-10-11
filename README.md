# Analytical AI Agent

An AI-powered analytical agent that processes CSV files using vector search and deterministic pandas operations.

## Features
- Vector-based semantic search using Gemini embeddings
- Deterministic pandas operations for numeric analysis
- LLM-driven natural language understanding
- Strict separation: LLM for understanding, pandas for computation

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Usage
See examples in `tests/` directory.
