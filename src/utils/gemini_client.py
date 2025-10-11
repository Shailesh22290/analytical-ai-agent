"""
Gemini API client for embeddings and text generation
"""
import google.generativeai as genai
from typing import List, Dict, Any
import numpy as np
from config.settings import settings


class GeminiClient:
    """Client for Google Gemini API"""
    
    def __init__(self):
        """Initialize Gemini client"""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not configured")
        
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
    
    def parse_intent(self, user_query: str, file_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse user query into structured action intent
        
        Args:
            user_query: Natural language query
            file_metadata: Metadata about available files
            
        Returns:
            Parsed intent as dictionary
        """
        # Build context about available files
        files_context = "\n".join([
            f"- File ID: {f['file_id']}, Columns: {', '.join(f['columns'])}, "
            f"Numeric columns: {', '.join(f['numeric_columns'])}"
            for f in file_metadata
        ])
        
        prompt = f"""You are an intent parser for an analytical agent. Parse the user query into a JSON action.

Available files:
{files_context}

Supported intents:
1. compare_averages - Compare average values of a column across files or groups
   Parameters: {{"column": str, "file1_id": str|null, "file2_id": str|null, "group_by": str|null}}

2. filter_threshold - Filter rows based on a numeric threshold
   Parameters: {{"column": str, "operator": str (>, <, >=, <=, ==), "value": float, "file_id": str|null}}

3. sort - Sort data by column
   Parameters: {{"column": str, "ascending": bool, "file_id": str|null, "limit": int|null}}

4. top_n - Get top N rows by column value
   Parameters: {{"column": str, "n": int, "ascending": bool, "file_id": str|null}}

5. compare_top - Compare top N items across two files
   Parameters: {{"column": str, "n": int, "file1_id": str|null, "file2_id": str|null}}

6. explain_row - Find and explain rows using semantic search
   Parameters: {{"query": str, "file_id": str|null, "top_k": int}}

User query: "{user_query}"

Parse this into JSON with keys "intent" and "parameters". If the query doesn't match any intent, return {{"intent": "unsupported", "parameters": {{}}}}.
Return ONLY valid JSON, no explanation."""

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
        
        return json.loads(text.strip())
    
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