"""
Main Analytical AI Agent
Orchestrates intent parsing, pandas analysis, and narrative generation
"""
from typing import Dict, Any, List, Optional
import json

from config.settings import settings
from src.utils.gemini_client import gemini_client
from src.utils.models import (
    AnalysisResult,
    ErrorResponse,
    CompareAveragesParams,
    FilterThresholdParams,
    SortParams,
    TopNParams,
    CompareTopParams,
    ExplainRowParams
)
from src.agents.ingestion import csv_ingestion
from src.agents.pandas_engine import pandas_engine
from src.vectordb.vector_store import vector_store_manager


class AnalyticalAgent:
    """Main agent for analytical queries"""
    
    def __init__(self):
        """Initialize agent"""
        self.supported_intents = settings.SUPPORTED_INTENTS
    
    def process_query(self, user_query: str, enhance_prompt: bool = False) -> Dict[str, Any]:
        """
        Process a user query end-to-end
        
        Args:
            user_query: Natural language query
            enhance_prompt: Whether to enhance the query first
            
        Returns:
            AnalysisResult or ErrorResponse as dict
        """
        try:
            # Step 1: Optional prompt enhancement
            if enhance_prompt:
                user_query = gemini_client.enhance_prompt(user_query)
                print(f"Enhanced query: {user_query}")
            
            # Step 2: Parse intent using LLM
            file_metadata_list = [
                meta.dict() for meta in csv_ingestion.file_metadata.values()
            ]
            
            if not file_metadata_list:
                return ErrorResponse(
                    error="no_data",
                    details="No CSV files have been loaded. Please ingest data first."
                ).dict()
            
            intent_data = gemini_client.parse_intent(user_query, file_metadata_list)
            
            intent = intent_data.get("intent")
            parameters = intent_data.get("parameters", {})
            
            # Check if intent is supported
            if intent not in self.supported_intents:
                return ErrorResponse(
                    error="unsupported_intent",
                    supported_intents=self.supported_intents,
                    details=f"Intent '{intent}' is not supported"
                ).dict()
            
            print(f"Parsed intent: {intent}")
            print(f"Parameters: {parameters}")
            
            # Step 3: Execute deterministic pandas analysis
            result_table, numbers = self._execute_intent(intent, parameters)
            
            # Step 4: Generate narrative using LLM
            narrative = gemini_client.generate_narrative(
                intent, 
                parameters, 
                result_table, 
                numbers
            )
            
            # Step 5: Return structured result
            result = AnalysisResult(
                result_table=result_table,
                numbers=numbers,
                narrative=narrative,
                metadata={
                    "intent": intent,
                    "parameters": parameters,
                    "query": user_query
                }
            )
            
            return result.dict()
            
        except Exception as e:
            return ErrorResponse(
                error="execution_error",
                details=str(e)
            ).dict()
    
    def _execute_intent(
        self, 
        intent: str, 
        parameters: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute the parsed intent using pandas engine
        
        Args:
            intent: Intent type
            parameters: Intent parameters
            
        Returns:
            (result_table, numbers)
        """
        if intent == "compare_averages":
            params = CompareAveragesParams(**parameters)
            return pandas_engine.compare_averages(params)
        
        elif intent == "filter_threshold":
            params = FilterThresholdParams(**parameters)
            return pandas_engine.filter_threshold(params)
        
        elif intent == "sort":
            params = SortParams(**parameters)
            return pandas_engine.sort_data(params)
        
        elif intent == "top_n":
            params = TopNParams(**parameters)
            return pandas_engine.top_n(params)
        
        elif intent == "compare_top":
            params = CompareTopParams(**parameters)
            return pandas_engine.compare_top(params)
        
        elif intent == "explain_row":
            return self._explain_row(ExplainRowParams(**parameters))
        
        else:
            raise ValueError(f"Unsupported intent: {intent}")
    
    def _explain_row(
        self, 
        params: ExplainRowParams
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Find and explain rows using semantic search
        
        Args:
            params: ExplainRowParams
            
        Returns:
            (result_table, numbers)
        """
        # Get file_id
        if params.file_id:
            file_id = params.file_id
        else:
            # Use first available file
            file_ids = list(csv_ingestion.dataframes.keys())
            if not file_ids:
                raise ValueError("No files loaded")
            file_id = file_ids[0]
        
        # Get vector store
        store = vector_store_manager.get_store(file_id)
        if not store:
            raise ValueError(f"Vector store for {file_id} not found")
        
        # Generate query embedding
        query_vector = gemini_client.generate_query_embedding(params.query)
        
        # Search for similar rows
        results = store.search(query_vector, k=params.top_k, file_id=file_id)
        
        # Get actual row data
        df = csv_ingestion.get_dataframe(file_id)
        result_table = []
        row_indices = []
        distances = []
        
        for meta, distance in results:
            if meta.is_row_vector:
                row_idx = meta.row_idx
                row_data = df.iloc[row_idx].to_dict()
                row_data['_row_index'] = row_idx
                row_data['_similarity_score'] = float(1 - distance)  # Convert distance to similarity
                result_table.append(row_data)
                row_indices.append(row_idx)
                distances.append(float(distance))
        
        numbers = {
            "query": params.query,
            "top_k": params.top_k,
            "row_indices": row_indices,
            "similarity_scores": [float(1 - d) for d in distances],
            "file_id": file_id
        }
        
        return result_table, numbers
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status and loaded data info"""
        files = csv_ingestion.list_files()
        vector_stores = vector_store_manager.list_stores()
        
        return {
            "status": "ready",
            "loaded_files": len(files),
            "files": files,
            "vector_stores": vector_stores,
            "supported_intents": self.supported_intents
        }


# Global agent instance
analytical_agent = AnalyticalAgent()