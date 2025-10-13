# Analytical AI Agent - Complete Installation & Usage Guide

## 📋 Project Overview

This is a production-ready Analytical AI Agent that:
- ✅ Ingests CSV files and creates vector embeddings
- ✅ Uses Gemini for intent parsing and narrative generation
- ✅ Executes **deterministic pandas operations** for all numeric computations
- ✅ Returns exact pandas outputs (no LLM hallucination of numbers)
- ✅ Supports semantic search via FAISS vector database

## 🚀 Quick Start (5 Minutes)

### Step 1: Clone/Create Project Structure

```bash
# Run the setup script
bash setup.sh

cd analytical-ai-agent
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Gemini API key
nano .env  # or use your favorite editor
```

Your `.env` file should contain:
```
GEMINI_API_KEY=your_actual_api_key_here
```

**Get your Gemini API key:**
1. Go to https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy and paste into `.env` file

### Step 5: Verify Installation

```bash
# Run tests (without API key needed for pandas tests)
pytest tests/test_agent.py -v

# Check agent status
python main.py status
```

## 📊 Usage Examples

### Example 1: Ingest CSV Data

```bash
# Ingest a single CSV file
python main.py ingest data/input/sales_q1.csv

# Ingest with custom file ID
python main.py ingest data/input/sales.csv --file-id my_sales

# Ingest without vectorization (faster, but no semantic search)
python main.py ingest data/input/data.csv --no-vectorize
```

### Example 2: Query the Agent

```bash
# Simple queries
python main.py query "What are the top 5 products by revenue?"

python main.py query "Show me products with price greater than 100"

python main.py query "What's the average rating?"

# Compare across files (after ingesting multiple CSVs)
python main.py query "Compare average sales between Q1 and Q2"

# Save results to JSON
python main.py query "Top 10 by revenue" --output results.json

# Enhance query for better clarity
python main.py query "give me expensive stuff" --enhance
```

### Example 3: Interactive Mode

```bash
python main.py interactive
```

Then type your queries:
```
>>> What are the top 5 products by price?
>>> Show products with rating above 4.5
>>> status
>>> quit
```

### Example 4: Programmatic Usage

```python
from src.agents.ingestion import csv_ingestion
from src.agents.analytical_agent import analytical_agent

# Ingest data
file_id, metadata = csv_ingestion.ingest_csv(
    "data/sales.csv",
    vectorize=True
)

# Query
result = analytical_agent.process_query(
    "What are the top 3 products by revenue?"
)

# Access results
print(result['narrative'])  # Human-readable text
print(result['numbers'])    # Exact computed numbers
print(result['result_table'])  # Raw pandas data
```

## 🎯 Supported Intents

The agent supports these analytical intents:

### 1. `compare_averages`
Compare average values across files or groups.

**Examples:**
- "Compare average price between file1 and file2"
- "What's the average revenue by category?"
- "Average sales in Q1 vs Q2"

### 2. `filter_threshold`
Filter rows based on numeric thresholds.

**Examples:**
- "Show products with price greater than 100"
- "Items with rating >= 4.5"
- "Revenue less than 1000"

### 3. `sort`
Sort data by column values.

**Examples:**
- "Sort by price descending"
- "Order by rating ascending"
- "Show top 10 sorted by revenue"

### 4. `top_n`
Get top N rows by column value.

**Examples:**
- "Top 5 products by revenue"
- "Best 10 rated items"
- "Highest 3 prices"

### 5. `compare_top`
Compare top N items across two files.

**Examples:**
- "Compare top 5 products Q1 vs Q2"
- "Top 3 revenue generators in both files"

### 6. `explain_row`
Find and explain rows using semantic search.

**Examples:**
- "Find products about laptops"
- "Show me electronic accessories"
- "Items related to furniture"

## 📁 Project Structure

```
analytical-ai-agent/
├── config/
│   └── settings.py          # Configuration settings
├── src/
│   ├── agents/
│   │   ├── ingestion.py     # CSV ingestion & vectorization
│   │   ├── pandas_engine.py # Deterministic pandas operations
│   │   └── analytical_agent.py  # Main agent orchestrator
│   ├── utils/
│   │   ├── gemini_client.py # Gemini API client
│   │   └── models.py        # Pydantic data models
│   └── vectordb/
│       └── vector_store.py  # FAISS vector database
├── tests/
│   └── test_agent.py        # Unit tests
├── examples/
│   └── example_usage.py     # Complete usage examples
├── data/
│   ├── input/               # Input CSV files
│   └── vectors/             # Stored vector databases
├── main.py                  # CLI interface
├── requirements.txt         # Dependencies
└── .env                     # Environment variables
```

## 🔧 Advanced Configuration

Edit `config/settings.py` to customize:

```python
# Change embedding model
EMBEDDING_MODEL = "models/embedding-001"

# Change generative model
GENERATIVE_MODEL = "gemini-2.0-flash-exp"

# Adjust vector database
VECTOR_DIMENSION = 768
FAISS_INDEX_TYPE = "Flat"  # or "IVFFlat" for large datasets

# LLM parameters
TEMPERATURE = 0.1  # Lower = more deterministic
MAX_TOKENS = 2048
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_agent.py::TestPandasEngine::test_top_n -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run example demonstration
python examples/example_usage.py
```

## 🎨 Response Format

All agent responses follow this JSON structure:

```json
{
  "result_table": [
    {"product": "Laptop", "price": 899.99, "revenue": 40499.55},
    {"product": "Monitor", "price": 299.99, "revenue": 23399.22}
  ],
  "numbers": {
    "n": 5,
    "column": "revenue",
    "top_values": [40499.55, 23399.22, ...],
    "average_of_top": 28500.45,
    "highest_value": 40499.55
  },
  "narrative": "Analysis identified the top 5 products by revenue. The highest revenue generator was Laptop at $40,499.55, with an average of $28,500.45 across the top 5.",
  "metadata": {
    "intent": "top_n",
    "parameters": {"column": "revenue", "n": 5},
    "query": "What are the top 5 products by revenue?"
  }
}
```

**Key Points:**
- `result_table`: Raw pandas data (exact values)
- `numbers`: All computed metrics (no LLM involvement)
- `narrative`: Human-readable explanation (LLM generated)
- `metadata`: Query tracking information

## 🔍 How It Works

### Architecture Overview

```
User Query
    ↓
[1] LLM: Parse Intent → JSON action
    ↓
[2] Pandas: Execute deterministic operations
    ↓
[3] Return: Exact numeric results
    ↓
[4] LLM: Generate narrative from results
    ↓
Final Response (with exact pandas outputs)
```

### The Three LLM Touchpoints

1. **Intent Parsing** (`gemini_client.parse_intent`)
   - Input: Natural language query
   - Output: Structured JSON action
   - Example: "top 5 by price" → `{"intent": "top_n", "parameters": {"column": "price", "n": 5}}`

2. **Prompt Enhancement** (`gemini_client.enhance_prompt`) - Optional
   - Input: Unclear query
   - Output: Clearer query
   - Example: "expensive stuff" → "products with highest prices"

3. **Narrative Generation** (`gemini_client.generate_narrative`)
   - Input: Exact pandas results
   - Output: Human-readable explanation
   - **Critical**: References only the provided numbers

### Deterministic Pandas Operations

All numeric computations in `pandas_engine.py`:
- ✅ Direct pandas DataFrame operations
- ✅ Returns exact values from data
- ✅ No LLM involvement in calculations
- ✅ Reproducible results

Example:
```python
# This is pandas, NOT LLM
avg = float(df[column].mean())  # Exact computation
top_df = df.nlargest(n, column)  # Exact sorting
filtered = df[df[column] > threshold]  # Exact filtering
```

## 📝 Example Queries

### Filtering & Thresholds

```bash
# Greater than
python main.py query "Show products with price > 200"

# Less than or equal
python main.py query "Items with rating <= 4.0"

# Equal to
python main.py query "Products priced at exactly 99.99"
```

### Sorting & Rankings

```bash
# Top N
python main.py query "Top 10 products by revenue"

# Bottom N
python main.py query "5 least expensive items"

# Sort ascending
python main.py query "Sort by price lowest to highest"
```

### Comparisons

```bash
# Compare averages
python main.py query "Average price in Q1 vs Q2"

# Compare top items
python main.py query "Compare top 5 sellers Q1 vs Q2"

# Group by comparison
python main.py query "Average revenue by category"
```

### Semantic Search

```bash
# Find similar items (requires vectorization)
python main.py query "Find products similar to laptops"

# Conceptual search
python main.py query "Show me office furniture items"
```

## ⚙️ Troubleshooting

### Issue: "GEMINI_API_KEY not set"

**Solution:**
```bash
# Make sure .env file exists
ls -la .env

# Check contents
cat .env

# Should show: GEMINI_API_KEY=your_key_here
```

### Issue: "No CSV files have been loaded"

**Solution:**
```bash
# Ingest data first
python main.py ingest data/input/yourfile.csv

# Check status
python main.py status
```

### Issue: "Vector store not found"

**Solution:**
```bash
# Re-ingest with vectorization
python main.py ingest data/input/yourfile.csv --file-id mydata

# Vectorization creates embeddings automatically
```

### Issue: Import errors

**Solution:**
```bash
# Make sure you're in the project root
pwd  # Should show: .../analytical-ai-agent

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: "Rate limit exceeded"

**Solution:**
- Gemini has free tier rate limits
- Wait a few minutes between large batches
- For vectorization of large CSVs, embeddings are created sequentially

## 🚦 Performance Tips

### For Large CSVs (10K+ rows)

1. **Disable vectorization initially:**
   ```bash
   python main.py ingest large_file.csv --no-vectorize
   ```

2. **Use sampling for testing:**
   ```python
   df = pd.read_csv("large.csv")
   df.head(1000).to_csv("sample.csv")
   ```

3. **Vectorize in batches:**
   - The ingestion module processes 50 rows at a time
   - Shows progress: "Processed 50/10000 rows"

### For Multiple Files

1. **Ingest all files first:**
   ```bash
   python main.py ingest file1.csv --file-id data1
   python main.py ingest file2.csv --file-id data2
   ```

2. **Vector stores are cached:**
   - Stored in `data/vectors/`
   - Automatically loaded when needed
   - No need to re-vectorize

## 🎓 Understanding the Design

### Why Separate LLM and Pandas?

**Problem:** LLMs can hallucinate numbers

**Solution:** 
- LLM only parses intent and generates text
- Pandas computes all actual numbers
- LLM narrative references pandas results

**Example:**
```
❌ BAD: LLM says "average is 42.5" (might be wrong)
✅ GOOD: Pandas computes avg = 42.7891
         LLM says "average is 42.79" (from pandas)
```

### Why Vector Database?

**Purpose:** Enable semantic search

**Use case:**
- Query: "Find laptop products"
- Without vectors: Need exact text match
- With vectors: Finds "Notebook PC", "Portable Computer", etc.

**When to skip:**
- Small datasets (< 100 rows)
- Only numeric queries (no text search)
- Fast ingestion needed

## 📚 API Reference

### CSVIngestion

```python
from src.agents.ingestion import csv_ingestion

# Ingest CSV
file_id, metadata = csv_ingestion.ingest_csv(
    filepath="data.csv",
    file_id="optional_custom_id",
    vectorize=True
)

# Get dataframe
df = csv_ingestion.get_dataframe(file_id)

# Get metadata
meta = csv_ingestion.get_metadata(file_id)

# List all files
files = csv_ingestion.list_files()
```

### PandasEngine

```python
from src.agents.pandas_engine import pandas_engine
from src.utils.models import TopNParams

# Execute operation
params = TopNParams(column="price", n=10, file_id="sales")
result_table, numbers = pandas_engine.top_n(params)
```

### AnalyticalAgent

```python
from src.agents.analytical_agent import analytical_agent

# Process query
result = analytical_agent.process_query(
    user_query="Top 5 by revenue",
    enhance_prompt=False
)

# Get status
status = analytical_agent.get_status()
```

## 🔐 Security Notes

- **API Key**: Never commit `.env` to git
- **Data**: CSV files in `data/input/` are gitignored
- **Vectors**: Vector databases in `data/vectors/` are gitignored

## 📄 License

This is a demonstration project. Modify as needed for your use case.

## 🤝 Contributing

To extend the agent:

1. **Add new intent:**
   - Update `SUPPORTED_INTENTS` in `settings.py`
   - Add parameters model in `models.py`
   - Implement in `pandas_engine.py`
   - Update `analytical_agent._execute_intent()`

2. **Add new data source:**
   - Extend `CSVIngestion` class
   - Support Excel, JSON, SQL, etc.

3. **Change vector database:**
   - Replace FAISS with Pinecone, Weaviate, etc.
   - Implement same interface in `vector_store.py`

## 🎉 You're Ready!

Start with:
```bash
# Run the complete demo
python examples/example_usage.py

# Then try interactive mode
python main.py interactive
```

For questions or issues, check the troubleshooting section above.