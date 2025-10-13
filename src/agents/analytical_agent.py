"""
Main Analytical AI Agent
Orchestrates intent parsing, pandas analysis, document queries, and narrative generation
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
    ExplainRowParams,
    DocumentQueryParams
)
from src.agents.ingestion import csv_ingestion
from src.agents.document_ingestion import document_ingestion
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
            
            # Step 2: Get all available data sources
            csv_metadata_list = [
                meta.dict() for meta in csv_ingestion.file_metadata.values()
            ]
            doc_metadata_list = [
                meta.dict() for meta in document_ingestion.document_metadata.values()
            ]
            
            if not csv_metadata_list and not doc_metadata_list:
                return ErrorResponse(
                    error="no_data",
                    details="No CSV or document files have been loaded. Please ingest data first."
                ).dict()
            
            # Step 3: Parse intent using LLM
            intent_data = gemini_client.parse_intent(
                user_query, 
                csv_metadata_list,
                doc_metadata_list
            )
            
            intent = intent_data.get("intent")
            parameters = intent_data.get("parameters", {})
            
            # Check if intent is supported
            if intent not in self.supported_intents:
                if settings.ENABLE_GENERAL_QUERY_FALLBACK:
                    intent = "general_query"
                    parameters = {"question": user_query}
                else:
                    return ErrorResponse(
                        error="unsupported_intent",
                        supported_intents=self.supported_intents,
                        details=f"Intent '{intent}' is not supported"
                    ).dict()
            
            print(f"Parsed intent: {intent}")
            print(f"Parameters: {parameters}")
            
            # Step 4: Execute based on intent type
            if intent == "general_query":
                return self._handle_general_query(user_query, parameters, csv_metadata_list, doc_metadata_list)
            elif intent == "document_query":
                return self._handle_document_query(user_query, parameters)
            else:
                # Execute deterministic pandas analysis
                result_table, numbers = self._execute_intent(intent, parameters)
                
                # Generate narrative using LLM
                narrative = gemini_client.generate_narrative(
                    intent, 
                    parameters, 
                    result_table, 
                    numbers
                )
                
                # Return structured result
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
            import traceback
            traceback.print_exc()
            return ErrorResponse(
                error="execution_error",
                details=str(e)
            ).dict()
    
    def _handle_general_query(
        self, 
        user_query: str,
        parameters: Dict[str, Any],
        csv_metadata_list: List[Dict[str, Any]],
        doc_metadata_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Handle general conversational queries about the data
        
        Args:
            user_query: Original user query
            parameters: Parsed parameters
            csv_metadata_list: List of CSV file metadata
            doc_metadata_list: List of document metadata
            
        Returns:
            Response dictionary
        """
        print(f"[DEBUG] Handling general query: {user_query}")
        
        # Get sample data if requested
        sample_data = {}
        file_id = parameters.get("file_id")
        
        try:
            if file_id:
                if file_id in csv_ingestion.dataframes:
                    df = csv_ingestion.get_dataframe(file_id)
                    sample_data[file_id] = df.head(3).to_dict('records')
            else:
                # Get samples from all CSV files
                for fid in list(csv_ingestion.dataframes.keys())[:2]:
                    df = csv_ingestion.get_dataframe(fid)
                    sample_data[fid] = df.head(3).to_dict('records')
        except Exception as e:
            print(f"[DEBUG] Could not get sample data: {e}")
        
        # Generate answer using LLM
        answer = gemini_client.answer_general_query(
            user_query,
            csv_metadata_list,
            doc_metadata_list,
            sample_data
        )
        
        # Format response
        result = {
            "result_table": [],
            "numbers": {
                "query_type": "general_information",
                "csv_files": len(csv_metadata_list),
                "document_files": len(doc_metadata_list),
                "total_rows": sum(m['num_rows'] for m in csv_metadata_list) if csv_metadata_list else 0
            },
            "narrative": answer,
            "metadata": {
                "intent": "general_query",
                "parameters": parameters,
                "query": user_query
            }
        }
        
        return result
    
    def _handle_document_query(
        self,
        user_query: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle document-specific queries (search within documents)
        
        Args:
            user_query: Original user query
            parameters: Parsed parameters including query and file_id
            
        Returns:
            Response dictionary
        """
        print(f"[DEBUG] Handling document query: {user_query}")
        
        params = DocumentQueryParams(**parameters)
        
        # Search documents
        search_results = document_ingestion.search_document(
            params.query,
            file_id=params.file_id,
            top_k=params.top_k
        )
        
        # Format results
        result_table = []
        for chunk_text, similarity, metadata in search_results:
            result_table.append({
                'file_id': metadata['file_id'],
                'chunk_type': metadata['chunk_type'],
                'similarity_score': f"{similarity:.4f}",
                'content': chunk_text,
                'question': metadata.get('question', ''),
                'answer': metadata.get('answer', ''),
                'analysis': metadata.get('analysis', '')
            })
        
        # Generate narrative using retrieved context
        context_text = "\n\n".join([
            f"[Relevance: {sim:.2f}] {chunk}" 
            for chunk, sim, _ in search_results
        ])
        
        narrative = gemini_client.answer_document_query(
            params.query,
            context_text,
            search_results
        )
        
        # Compute numbers
        numbers = {
            "query": params.query,
            "num_results": len(search_results),
            "avg_similarity": sum(s for _, s, _ in search_results) / len(search_results) if search_results else 0,
            "file_ids": list(set(m['file_id'] for _, _, m in search_results))
        }
        
        result = {
            "result_table": result_table,
            "numbers": numbers,
            "narrative": narrative,
            "metadata": {
                "intent": "document_query",
                "parameters": parameters,
                "query": user_query
            }
        }
        
        return result
    
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
                row_data['_similarity_score'] = float(1 - distance)
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
        csv_files = csv_ingestion.list_files()
        doc_files = document_ingestion.list_documents()
        vector_stores = vector_store_manager.list_stores()
        
        return {
            "status": "ready",
            "loaded_csv_files": len(csv_files),
            "loaded_documents": len(doc_files),
            "csv_files": csv_files,
            "documents": doc_files,
            "vector_stores": vector_stores,
            "supported_intents": self.supported_intents
        }


# Global agent instance
analytical_agent = AnalyticalAgent()