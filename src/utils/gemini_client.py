"""
Gemini API client for embeddings and text generation
Extended with document query support
"""
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import numpy as np
from config.settings import settings
import logging

logger = logging.getLogger(__name__) 

class GeminiClient:
    """Client for Google Gemini API"""
    
    def __init__(self):
        """Initialize Gemini client"""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not configured")
        try:
            key_preview = settings.GEMINI_API_KEY[-4:]
            logger.info(f"Configuring Gemini client with API key ending in: ...{key_preview}")
        except Exception:
            logger.warning("Could not display API key preview.")
            
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.embedding_model = settings.EMBEDDING_MODEL
        self.generative_model = genai.GenerativeModel(settings.GENERATIVE_MODEL)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_document"
        )
        return np.array(result['embedding'], dtype=np.float32)
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            Matrix of embeddings (n_texts x embedding_dim)
        """
        embeddings = []
        
        # Process in batches to avoid rate limits
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                emb = self.generate_embedding(text)
                embeddings.append(emb)
        
        return np.vstack(embeddings)
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        result = genai.embed_content(
            model=self.embedding_model,
            content=query,
            task_type="retrieval_query"
        )
        return np.array(result['embedding'], dtype=np.float32)
    
    def parse_intent(
        self, 
        user_query: str, 
        csv_metadata: List[Dict[str, Any]],
        doc_metadata: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Parse user query into structured action intent
        
        Args:
            user_query: Natural language query
            csv_metadata: Metadata about available CSV files
            doc_metadata: Metadata about available analysis documents
            
        Returns:
            Parsed intent as dictionary
        """
        # Build context about available files
        csv_context = "\n".join([
            f"- CSV File ID: {f['file_id']}, Columns: {', '.join(f['columns'])}, "
            f"Numeric columns: {', '.join(f['numeric_columns'])}"
            for f in csv_metadata
        ]) if csv_metadata else "No CSV files loaded"
        
        doc_context = "\n".join([
            f"- Analysis ID: {d['file_id']}, Type: {d['document_type']}, "
            f"Filename: {d['filename']}, Q&A pairs: {d.get('num_qa_pairs', 0)}"
            for d in (doc_metadata or [])
        ]) if doc_metadata else "No analysis documents loaded"
        
        prompt = f"""You are an intent parser for an analytical agent. Parse the user query into a JSON action.

Available CSV files:
{csv_context}

Available Analysis Documents:
{doc_context}

CRITICAL INSTRUCTIONS:
- If query asks about ANALYSIS CONTENT (questions from analysis, explain bearing diagnostics, analysis insights), use document_query (8)
- If query asks for CSV DATA COMPUTATION (top, highest, lowest, filter, sort, average), use analytical intents (1-6)
- If query asks for DESCRIPTION of data structure, use general_query (7)

Supported intents:

ANALYTICAL INTENTS (for CSV computations):
1. top_n - Get top/bottom N rows by value
   Use when: "top 5", "highest 10", "lowest 3", "show me the maximum", "find the minimum"
   Parameters: {{"column": str, "n": int, "ascending": bool, "file_id": str|null}}

2. filter_threshold - Filter rows by numeric condition
   Use when: "where", "greater than", "less than", "filter", "rows above/below"
   Parameters: {{"column": str, "operator": str (>, <, >=, <=, ==), "value": float, "file_id": str|null}}

3. compare_averages - Calculate and compare averages
   Use when: "average", "mean", "compare average"
   Parameters: {{"column": str, "file1_id": str|null, "file2_id": str|null, "group_by": str|null}}

4. sort - Sort data by column
   Use when: "sort", "order by", "arrange by"
   Parameters: {{"column": str, "ascending": bool, "file_id": str|null, "limit": int|null}}

5. compare_top - Compare top N across files
   Use when: "compare top", "compare highest between files"
   Parameters: {{"column": str, "n": int, "file1_id": str|null, "file2_id": str|null}}

6. explain_row - Semantic search for CSV rows
   Use when: "find rows about", "search for", "show items related to"
   Parameters: {{"query": str, "file_id": str|null, "top_k": int}}

DESCRIPTIVE INTENT (for information):
7. general_query - Answer questions about data structure
   Use when: "describe", "what columns", "explain structure", "tell me about files", "list files"
   Parameters: {{"question": str, "file_id": str|null}}

ANALYSIS INTENT (for analysis content):
8. document_query - Answer questions from analysis content
   Use when: "what does the analysis say", "explain from analysis", "bearing diagnostics analysis", "Q1 from analysis", "rise in envelope", "analysis of kurtosis"
   Parameters: {{"query": str, "file_id": str|null, "top_k": int}}

User query: "{user_query}"

DECISION RULES:
- "highest/lowest VALUE in CSV" → top_n (analytical)
- "what does analysis say about X" → document_query (analysis)
- "describe CSV structure" → general_query (descriptive)
- "rise in envelope/acceleration/velocity" → document_query (analysis)
- "harmonic energy", "kurtosis", "crest factor" → document_query (analysis)

Parse this into JSON with keys "intent" and "parameters". 
Return ONLY valid JSON, no explanation.

Examples:
- "Show the highest value" → {{"intent": "top_n", "parameters": {{"column": "auto-detect", "n": 1, "ascending": false}}}}
- "What does the analysis say about envelope rise?" → {{"intent": "document_query", "parameters": {{"query": "envelope rise", "top_k": 3}}}}
- "Describe the CSV" → {{"intent": "general_query", "parameters": {{"question": "Describe the CSV"}}}}
"""

        response = self.generative_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=settings.TEMPERATURE,
                max_output_tokens=settings.MAX_TOKENS
            )
        )
        
        # Extract JSON from response
        import json
        text = response.text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        parsed = json.loads(text.strip())
        
        # Post-process: If top_n with auto-detect column, find numeric columns
        if parsed.get('intent') == 'top_n' and parsed.get('parameters', {}).get('column') == 'auto-detect':
            if csv_metadata and csv_metadata[0].get('numeric_columns'):
                numeric_cols = csv_metadata[0]['numeric_columns']
                parsed['parameters']['column'] = numeric_cols[0] if numeric_cols else 'value'
        
        return parsed
    
    def answer_general_query(
        self,
        question: str,
        csv_metadata: List[Dict[str, Any]],
        doc_metadata: List[Dict[str, Any]] = None,
        sample_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Answer general conversational queries about the data
        
        Args:
            question: User's question
            csv_metadata: Metadata about loaded CSV files
            doc_metadata: Metadata about loaded analysis documents
            sample_data: Optional sample data from files
            
        Returns:
            Natural language answer
        """
        # Build comprehensive context
        csv_info = []
        for meta in (csv_metadata or []):
            info = f"""
CSV File: {meta.get('filename', meta['file_id'])}
- Rows: {meta['num_rows']}
- Columns: {meta['num_columns']}
- Column names: {', '.join(meta['columns'])}
- Numeric columns: {', '.join(meta['numeric_columns'])}
"""
            csv_info.append(info)
        
        doc_info = []
        for meta in (doc_metadata or []):
            info = f"""
Analysis Document: {meta.get('filename', meta['file_id'])}
- Type: {meta.get('document_type', 'unknown')}
- Characters: {meta.get('num_characters', 0)}
- Q&A pairs: {meta.get('num_qa_pairs', 0)}
"""
            doc_info.append(info)
        
        csv_context = "\n".join(csv_info) if csv_info else "No CSV files loaded"
        doc_context = "\n".join(doc_info) if doc_info else "No analysis documents loaded"
        
        # Add sample data if provided
        sample_context = ""
        if sample_data:
            sample_context = f"\n\nSample CSV data:\n{sample_data}"
        
        prompt = f"""You are a helpful data analyst assistant. Answer the user's question about their data files.

Available CSV Files:
{csv_context}

Available Analysis Documents:
{doc_context}{sample_context}

User Question: "{question}"

Provide a clear, informative answer. If the question asks about:
- File structure: Describe columns, data types, row counts
- Data content: Explain what kind of data is present
- Available analyses: Suggest what queries they can run
- Column details: Explain specific columns if asked
- Analysis documents: Mention available analysis documents and their content

Be conversational and helpful. Format your response with clear sections if needed."""

        response = self.generative_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1000
            )
        )
        
        return response.text.strip()
    
    def answer_document_query(
        self,
        query: str,
        context_text: str,
        search_results: List[tuple]
    ) -> str:
        """
        Answer query using retrieved analysis context
        
        Args:
            query: User's question
            context_text: Retrieved context from analysis
            search_results: List of (chunk, similarity, metadata) tuples
            
        Returns:
            Natural language answer
        """
        # Check if we have Q&A pairs in results
        qa_pairs = []
        for _, _, meta in search_results:
            if meta.get('question') and meta.get('answer'):
                qa_pairs.append({
                    'question': meta['question'],
                    'answer': meta['answer'],
                    'analysis': meta.get('analysis', '')
                })
        
        qa_context = ""
        if qa_pairs:
            qa_context = "\n\nRelevant Q&A pairs from analysis:\n" + "\n\n".join([
                f"Q: {qa['question']}\nA: {qa['answer']}" + 
                (f"\nAnalysis: {qa['analysis']}" if qa['analysis'] else "")
                for qa in qa_pairs[:3]
            ])
        
        prompt = f"""You are an expert bearing diagnostics analyst. Answer the user's question using the provided analysis context.

User Question: "{query}"

Retrieved Context:
{context_text}{qa_context}

Instructions:
- Answer directly and specifically based on the context provided
- If the query asks about comparisons (current vs best), calculate percentage changes
- If the query asks about analysis, provide the analysis from the provided material
- Use technical terms appropriately (envelope, acceleration, velocity, kurtosis, crest factor, harmonic energy, BPFI, BPFO, BSF, FTF)
- Structure your answer clearly with observations and analysis sections if relevant
- If context doesn't contain exact answer, say so and provide what's available

Provide a comprehensive, professional answer according to the analysis."""

        response = self.generative_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1500
            )
        )
        
        return response.text.strip()
    
    def generate_narrative(
        self, 
        intent: str, 
        parameters: Dict[str, Any],
        result_table: List[Dict[str, Any]],
        numbers: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable narrative from results
        
        Args:
            intent: The action intent
            parameters: Action parameters
            result_table: Raw result data
            numbers: Computed numbers
            
        Returns:
            Natural language narrative
        """
        prompt = f"""Generate a clear, concise narrative explanation of these analysis results.

Intent: {intent}
Parameters: {parameters}

Computed numbers: {numbers}

Result preview (first 5 rows): {result_table[:5]}

Write a narrative that:
1. Explains what analysis was performed
2. Highlights key findings from the numbers
3. References specific values from the computed numbers
4. Is 2-4 sentences long
5. Is professional and clear

Generate only the narrative text, no preamble."""

        response = self.generative_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=500
            )
        )
        
        return response.text.strip()
    
    def enhance_prompt(self, user_query: str) -> str:
        """
        Enhance user prompt for better clarity
        
        Args:
            user_query: Original user query
            
        Returns:
            Enhanced query
        """
        prompt = f"""Rephrase this analytical query to be more precise and clear while preserving the original intent:

"{user_query}"

Return only the rephrased query, no explanation."""

        response = self.generative_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=200
            )
        )
        
        return response.text.strip()


# Global client instance
gemini_client = GeminiClient()