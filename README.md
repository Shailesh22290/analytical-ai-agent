
# Analytical AI Agent

An AI-powered data analysis system that understands natural language questions and performs **accurate, hallucination-free analysis** using pandas — while generating human-friendly insights with an LLM.

> The LLM **never touches numbers** — all calculations are done by pandas to ensure correctness.

---

##  Features

| Feature                       | Description                                         |
| ----------------------------- | --------------------------------------------------- |
| 🧠 Natural-language analysis  | Ask questions like *"Show top 5 products by sales"* |
| 📉 Zero numeric hallucination | All numbers computed by pandas, not LLM             |
| 📁 Smart CSV ingestion        | Detect columns, extract metadata, store securely    |
| 🔍 FAISS vector search        | Semantic row lookup & context retrieval             |
| 🧾 Structured output          | Raw results + exact numbers + narrative summary     |
| 🛠 Modular architecture       | LLM layer + pandas engine + vector DB               |
| 🧪 Fully tested               | Unit & integration tests included                   |

---

## 🏗 Architecture Overview

```
User Question → LLM Interprets Intent → Pandas Executes Query
            → LLM Writes Explanation → Final Structured Output
```

### Why this design?

✅ LLM handles language & explanations
✅ pandas handles all math, filters, sorting
✅ FAISS handles semantic search
❌ LLM never fabricates numbers

---

## 📂 Folder Structure

```
analytical-ai-agent/
├── src/
│   ├── agents/ (core logic)
│   ├── utils/ (LLM wrapper, schemas)
│   └── vectordb/ (FAISS store)
├── data/ (CSV + vector indexes)
├── examples/
├── app.py 
└── README.md
```

---

## 🚀 How It Works

### Example Query

> **"Show the top 5 products by revenue"**

**Step 1:** LLM → Parse intent
**Step 2:** pandas → Compute exact result
**Step 3:** LLM → Explain in plain English

**Output Format:**

```json
{
  "result_table": [...],   // Raw dataframe rows
  "numbers": {...},        // Exact computed values
  "narrative": "..."       // Detailed explanation
}
```

---

## 🧠 Supported Operations

| Task                | Example                     |
| ------------------- | --------------------------- |
| Top N results       | "Show top 10 by profit"     |
| Column filters      | "Sales above 1000"          |
| Compare averages    | "Compare Q1 vs Q2 revenue"  |
| Semantic row search | "find laptop-related items" |

---

##  Quick Start

```bash
git clone https://github.com/Shailesh22290/analytical-ai-agent
cd analytical-ai-agent
pip install -r requirements.txt
```

### Run demo

```bash
streamlit run app.py
```


---

## 💡 Future Enhancements

* Excel support
* Charts & dashboards
* Time-series analytics
* FastAPI interface for production use
* Async embeddings for large CSVs

---

## 🎯 Ideal Use Cases

| Use Case                    | Example                                |
| --------------------------- | -------------------------------------- |
| Business analytics          | Sales, finance, operations             |
| Student / research analysis | CSV-based research papers              |
| Internal BI tools           | Private analysis without exposing data |
| No-code data querying       | Analysts who hate SQL 😄               |

---

## 👏 Credits

Built with:

* **pandas** for analysis
* **FAISS** for semantic search
* **Gemini API** for language understanding

---
