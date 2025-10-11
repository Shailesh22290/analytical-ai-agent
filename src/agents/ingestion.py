"""
CSV ingestion and vectorization module
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import hashlib

from src.utils.gemini_client import gemini_client
from src.utils.models import FileMetadata, VectorMetadata
from src.vectordb.vector_store import vector_store_manager


class CSVIngestion:
    """Handles CSV file ingestion and vectorization"""
    
    def __init__(self):
        """Initialize ingestion handler"""
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.file_metadata: Dict[str, FileMetadata] = {}
    
    def generate_file_id(self, filename: str) -> str:
        """
        Generate unique file ID from filename
        
        Args:
            filename: Original filename
            
        Returns:
            Unique file identifier
        """
        # Create hash from filename and timestamp
        unique_str = f"{filename}_{datetime.now().isoformat()}"
        file_hash = hashlib.md5(unique_str.encode()).hexdigest()[:8]
        clean_name = Path(filename).stem.replace(" ", "_")
        return f"{clean_name}_{file_hash}"
    
    def analyze_dataframe(self, df: pd.DataFrame, file_id: str, filename: str) -> FileMetadata:
        """
        Analyze dataframe and extract metadata
        
        Args:
            df: Input dataframe
            file_id: File identifier
            filename: Original filename
            
        Returns:
            FileMetadata object
        """
        # Get column types
        column_types = {}
        numeric_columns = []
        text_columns = []
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            column_types[col] = dtype
            
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col)
            else:
                text_columns.append(col)
        
        metadata = FileMetadata(
            file_id=file_id,
            filename=filename,
            num_rows=len(df),
            num_columns=len(df.columns),
            columns=list(df.columns),
            column_types=column_types,
            numeric_columns=numeric_columns,
            text_columns=text_columns,
            ingestion_timestamp=datetime.now().isoformat()
        )
        
        return metadata
    
    def create_row_text(self, row: pd.Series, columns: List[str]) -> str:
        """
        Create text representation of a row for embedding
        
        Args:
            row: Pandas Series (row)
            columns: Column names
            
        Returns:
            Text representation
        """
        parts = []
        for col in columns:
            value = row[col]
            if pd.notna(value):
                parts.append(f"{col}: {value}")
        
        return " | ".join(parts)
    
    def create_column_summary(self, df: pd.DataFrame, column: str) -> str:
        """
        Create summary text for a column
        
        Args:
            df: Dataframe
            column: Column name
            
        Returns:
            Summary text
        """
        col_data = df[column]
        
        if pd.api.types.is_numeric_dtype(col_data):
            summary = (
                f"Column {column}: numeric. "
                f"Min: {col_data.min()}, Max: {col_data.max()}, "
                f"Mean: {col_data.mean():.2f}, Median: {col_data.median()}"
            )
        else:
            unique_vals = col_data.nunique()
            top_vals = col_data.value_counts().head(3)
            summary = (
                f"Column {column}: text/categorical. "
                f"Unique values: {unique_vals}. "
                f"Top values: {', '.join(str(v) for v in top_vals.index)}"
            )
        
        return summary
    
    def ingest_csv(
        self, 
        filepath: str, 
        file_id: Optional[str] = None,
        vectorize: bool = True
    ) -> Tuple[str, FileMetadata]:
        """
        Ingest CSV file and optionally vectorize
        
        Args:
            filepath: Path to CSV file
            file_id: Optional custom file ID
            vectorize: Whether to create embeddings
            
        Returns:
            Tuple of (file_id, metadata)
        """
        # Read CSV
        df = pd.read_csv(filepath)
        
        # Generate file ID if not provided
        if file_id is None:
            file_id = self.generate_file_id(Path(filepath).name)
        
        # Analyze dataframe
        metadata = self.analyze_dataframe(df, file_id, Path(filepath).name)
        
        # Store dataframe and metadata
        self.dataframes[file_id] = df
        self.file_metadata[file_id] = metadata
        
        print(f"✓ Loaded {filepath}: {len(df)} rows, {len(df.columns)} columns")
        
        # Vectorize if requested
        if vectorize:
            self._vectorize_dataframe(df, file_id, metadata)
        
        return file_id, metadata
    
    def _vectorize_dataframe(
        self, 
        df: pd.DataFrame, 
        file_id: str,
        metadata: FileMetadata
    ) -> None:
        """
        Create embeddings for dataframe rows and columns
        
        Args:
            df: Dataframe to vectorize
            file_id: File identifier
            metadata: File metadata
        """
        print(f"Creating embeddings for {file_id}...")
        
        # Create vector store
        store = vector_store_manager.create_store(file_id)
        
        vectors_list = []
        metadata_list = []
        
        # 1. Vectorize each row
        print(f"  - Vectorizing {len(df)} rows...")
        for idx, row in df.iterrows():
            row_text = self.create_row_text(row, df.columns)
            
            # Generate embedding
            embedding = gemini_client.generate_embedding(row_text)
            vectors_list.append(embedding)
            
            # Create metadata
            vec_meta = VectorMetadata(
                file_id=file_id,
                row_idx=int(idx),
                column_name=None,
                is_row_vector=True,
                original_text=row_text[:500]  # Store first 500 chars
            )
            metadata_list.append(vec_meta)
            
            if (idx + 1) % 50 == 0:
                print(f"    Processed {idx + 1}/{len(df)} rows")
        
        # 2. Vectorize column summaries
        print(f"  - Vectorizing {len(df.columns)} column summaries...")
        for col in df.columns:
            col_summary = self.create_column_summary(df, col)
            
            # Generate embedding
            embedding = gemini_client.generate_embedding(col_summary)
            vectors_list.append(embedding)
            
            # Create metadata
            vec_meta = VectorMetadata(
                file_id=file_id,
                row_idx=-1,  # -1 indicates column summary
                column_name=col,
                is_row_vector=False,
                original_text=col_summary
            )
            metadata_list.append(vec_meta)
        
        # Add all vectors to store
        vectors_array = np.vstack(vectors_list)
        store.add_vectors(vectors_array, metadata_list)
        
        # Save to disk
        vector_store_manager.save_store(file_id)
        
        print(f"✓ Created {len(vectors_list)} embeddings for {file_id}")
    
    def get_dataframe(self, file_id: str) -> pd.DataFrame:
        """Get dataframe by file_id"""
        if file_id not in self.dataframes:
            raise ValueError(f"File {file_id} not loaded")
        return self.dataframes[file_id]
    
    def get_metadata(self, file_id: str) -> FileMetadata:
        """Get metadata by file_id"""
        if file_id not in self.file_metadata:
            raise ValueError(f"Metadata for {file_id} not found")
        return self.file_metadata[file_id]
    
    def list_files(self) -> List[Dict[str, any]]:
        """List all loaded files"""
        return [
            {
                "file_id": fid,
                "filename": meta.filename,
                "rows": meta.num_rows,
                "columns": meta.num_columns,
                "numeric_columns": meta.numeric_columns
            }
            for fid, meta in self.file_metadata.items()
        ]


# Global ingestion instance
csv_ingestion = CSVIngestion()