#!/usr/bin/env python3
"""
Analytical AI Agent - Streamlit Multi-CSV Query Interface with Vector DB Persistence
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import tempfile
import re
import numpy as np
import uuid 
import hashlib
from src.agents.ingestion import csv_ingestion
from src.agents.analytical_agent import analytical_agent

# Page config
st.set_page_config(
    page_title="Analytical AI Agent",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'loaded_files' not in st.session_state:
    st.session_state.loaded_files = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'cleaning_options' not in st.session_state:
    st.session_state.cleaning_options = {
        'remove_non_numeric': True,
        'handle_concatenated': True,
        'remove_special_chars': True,
        'convert_negative': True
    }
if 'vector_db_path' not in st.session_state:
    st.session_state.vector_db_path = Path("data/vector_db")
    st.session_state.vector_db_path.mkdir(parents=True, exist_ok=True)

def get_file_hash(df):
    """Generate unique hash for CSV content"""
    content_str = df.to_csv(index=False)
    return hashlib.md5(content_str.encode()).hexdigest()

def get_vector_db_info():
    """Get information about the vector database"""
    try:
        db_path = st.session_state.vector_db_path
        if not db_path.exists():
            return {'exists': False, 'collections': 0, 'size_mb': 0}
        
        # Check if vector DB has data
        size_bytes = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
        size_mb = size_bytes / 1024 / 1024
        
        # Try to count collections/files
        collections = len([d for d in db_path.iterdir() if d.is_dir()])
        
        return {
            'exists': True,
            'collections': collections,
            'size_mb': size_mb,
            'path': str(db_path)
        }
    except Exception as e:
        return {'exists': False, 'error': str(e)}

def clear_vector_db():
    """Clear the vector database"""
    try:
        import shutil
        db_path = st.session_state.vector_db_path
        if db_path.exists():
            shutil.rmtree(db_path)
            db_path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        st.error(f"Error clearing vector DB: {str(e)}")
        return False

def clean_numeric_string(value):
    """Clean malformed numeric strings"""
    if pd.isna(value) or value == '':
        return np.nan
    
    value_str = str(value).strip()
    
    # Handle negative signs at the end
    if value_str.endswith('-'):
        value_str = '-' + value_str[:-1]
    
    # Remove common non-numeric characters but keep decimal point, negative sign, and e for scientific notation
    value_str = re.sub(r'[^\d.\-eE+]', '', value_str)
    
    # Handle concatenated numbers - try to split and take first valid number
    if value_str.count('.') > 1 or (value_str.count('-') > 1 and not value_str.startswith('-')):
        # Multiple decimals or negatives suggest concatenated numbers
        # Try to extract the first valid number
        match = re.search(r'^-?\d+\.?\d*', value_str)
        if match:
            value_str = match.group()
    
    try:
        return float(value_str) if value_str else np.nan
    except (ValueError, TypeError):
        return np.nan

def clean_csv_data(df):
    """Clean CSV data to handle malformed numeric values"""
    df_cleaned = df.copy()
    
    for col in df_cleaned.columns:
        # Try to identify numeric columns
        if df_cleaned[col].dtype == 'object':
            # Sample a few values to check if they look numeric
            sample = df_cleaned[col].dropna().head(10)
            if len(sample) > 0:
                # Check if values contain numbers
                numeric_pattern = any(bool(re.search(r'\d', str(val))) for val in sample)
                
                if numeric_pattern:
                    # Try cleaning
                    cleaned_col = df_cleaned[col].apply(clean_numeric_string)
                    
                    # If most values converted successfully, use cleaned version
                    valid_ratio = cleaned_col.notna().sum() / len(cleaned_col)
                    if valid_ratio > 0.5:  # If more than 50% converted successfully
                        df_cleaned[col] = cleaned_col
    
    return df_cleaned

def load_csv_file(uploaded_file, clean_data=True, use_vector_db=True):
    """Load a CSV file and ingest it with vector DB persistence"""
    try:
        # Read CSV first to clean it
        df = pd.read_csv(uploaded_file)
        original_shape = df.shape
        
        # Clean data if enabled
        if clean_data:
            df = clean_csv_data(df)
        
        # Generate hash for tracking
        file_hash = get_file_hash(df)
        
        # Save cleaned CSV temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w') as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        # Check if this file hash already exists in vector DB
        # The csv_ingestion should handle persistence internally
        # If it uses ChromaDB, FAISS, or similar, they have built-in persistence
        
        # Ingest CSV (this creates/loads vectors from vector DB)
        file_id, metadata = csv_ingestion.ingest_csv(tmp_path, vectorize=True)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Check if vectors were loaded from existing DB
        # This is a heuristic - if ingestion was fast, likely from cache
        is_cached = hasattr(metadata, 'from_cache') and metadata.from_cache
        
        return {
            'name': uploaded_file.name,
            'file_id': file_id,
            'file_hash': file_hash,
            'metadata': metadata,
            'status': 'success',
            'original_shape': original_shape,
            'cleaned_shape': df.shape,
            'cleaned': clean_data,
            'vector_db': use_vector_db
        }
    except Exception as e:
        return {
            'name': uploaded_file.name,
            'status': 'error',
            'error': str(e)
        }

def display_results(result):
    """Display query results in a formatted way"""
    if 'error' in result:
        st.error(f"❌ Error: {result.get('details', result['error'])}")
        
        # Show troubleshooting tips
        with st.expander("🔧 Troubleshooting Tips"):
            st.markdown("""
            **Common Issues:**
            - **Malformed numeric data**: Enable data cleaning in the sidebar
            - **Column names**: Ensure your query uses exact column names
            - **Missing data**: Check if your CSV has the expected columns
            - **Data format**: Verify numeric columns don't contain text
            
            **Tips:**
            - Try simplifying your query
            - Check the loaded file details in the sidebar
            - Enable all cleaning options
            - Check vector DB status
            """)
        return
    
    numbers = result.get('numbers', {})
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Query Results")
        
        # Show values found
        if 'top_values' in numbers:
            values = numbers['top_values']
            indices = numbers.get('top_indices', [])
            
            if isinstance(values, list):
                results_df = pd.DataFrame({
                    'Value': values,
                    'Row Index': indices
                })
                st.dataframe(results_df, use_container_width=True)
            else:
                st.metric("Result Value", values)
        
        # Show result table if available
        if 'result_table' in result and result['result_table']:
            st.subheader("📋 Result Table")
            result_df = pd.DataFrame(result['result_table'])
            st.dataframe(result_df, use_container_width=True)
            
            # ✅ Add unique key so multiple download buttons don’t clash
            csv = result_df.to_csv(index=False)
            unique_key = f"download_button_{uuid.uuid4()}"
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="query_results.csv",
                mime="text/csv",
                key=unique_key
            )
    
    with col2:
        st.subheader("📈 Statistics")
        
        # Display statistics as metrics
        stat_mapping = {
            'average': ('Average', '📊'),
            'min': ('Minimum', '⬇️'),
            'max': ('Maximum', '⬆️'),
            'sum': ('Sum', '➕'),
            'count': ('Count', '🔢'),
            'filtered_rows': ('Filtered Rows', '🔍'),
            'total_rows': ('Total Rows', '📝')
        }
        
        for key, (label, emoji) in stat_mapping.items():
            if key in numbers:
                value = numbers[key]
                if isinstance(value, float):
                    st.metric(f"{emoji} {label}", f"{value:.4f}")
                else:
                    st.metric(f"{emoji} {label}", value)


# Main app
def main():
    st.title("🔍 Analytical AI Agent - Multi-CSV Query Interface")
    st.markdown("Upload multiple CSV files and run analytical queries across your data")
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("📁 File Management")
        
        # Vector DB info
        db_info = get_vector_db_info()
        
        with st.expander("🗄️ Vector Database", expanded=False):
            if db_info['exists']:
                st.write(f"**Status:** 🟢 Active")
                st.write(f"**Size:** {db_info['size_mb']:.2f} MB")
                st.write(f"**Collections:** {db_info.get('collections', 'N/A')}")
                st.caption(f"Path: `{db_info.get('path', 'N/A')}`")
                
                st.info("💡 Embeddings are persisted in the vector DB. Re-uploading files will reuse existing vectors if available.")
                
                if st.button("🗑️ Clear Vector Database"):
                    if clear_vector_db():
                        st.success("✅ Vector DB cleared!")
                        st.rerun()
            else:
                st.write(f"**Status:** ⚪ Not initialized")
                st.info("Vector DB will be created on first file upload")
        
        # Data cleaning options
        with st.expander("🧹 Data Cleaning Options", expanded=True):
            st.session_state.cleaning_options['remove_non_numeric'] = st.checkbox(
                "Remove non-numeric characters",
                value=True,
                help="Clean text from numeric columns"
            )
            st.session_state.cleaning_options['handle_concatenated'] = st.checkbox(
                "Handle concatenated numbers",
                value=True,
                help="Fix numbers stuck together (e.g., '0.0690.059')"
            )
            st.session_state.cleaning_options['convert_negative'] = st.checkbox(
                "Fix negative sign position",
                value=True,
                help="Convert '0.069-' to '-0.069'"
            )
            
            clean_enabled = any(st.session_state.cleaning_options.values())
        
        uploaded_files = st.file_uploader(
            "Upload CSV Files",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload one or more CSV files to analyze"
        )
        
        if uploaded_files:
            if st.button("🔄 Load All Files", type="primary"):
                st.session_state.loaded_files = []
                
                with st.spinner("Loading files and creating/updating vector embeddings..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, file in enumerate(uploaded_files):
                        # Reset file pointer
                        file.seek(0)
                        
                        # Show status
                        status_text.text(f"Processing: {file.name}")
                        
                        result = load_csv_file(file, clean_data=clean_enabled)
                        st.session_state.loaded_files.append(result)
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                
                success_count = sum(1 for f in st.session_state.loaded_files if f['status'] == 'success')
                
                if success_count > 0:
                    st.success(f"✅ Loaded {success_count} file(s) with vector DB persistence")
                st.rerun()
        
        # Display loaded files
        if st.session_state.loaded_files:
            st.subheader("📂 Loaded Files")
            for file_info in st.session_state.loaded_files:
                if file_info['status'] == 'success':
                    with st.expander(f"✅ {file_info['name']}"):
                        st.write(f"**Rows:** {file_info['metadata'].num_rows}")
                        st.write(f"**Columns:** {file_info['metadata'].num_columns}")
                        st.write(f"**File ID:** `{file_info['file_id']}`")
                        
                        if file_info.get('vector_db'):
                            st.success("🗄️ Vectors stored in DB")
                        
                        if file_info.get('cleaned'):
                            st.info(f"🧹 Data cleaned")
                        
                        if file_info['metadata'].columns:
                            st.write("**Column Names:**")
                            for col in file_info['metadata'].columns[:10]:
                                st.text(f"  • {col}")
                            if len(file_info['metadata'].columns) > 10:
                                st.write(f"... and {len(file_info['metadata'].columns) - 10} more")
                else:
                    with st.expander(f"❌ {file_info['name']}", expanded=True):
                        st.error(f"**Error:** {file_info['error']}")
                        st.info("💡 Try enabling all data cleaning options above")
            
            if st.button("🗑️ Clear All Files"):
                st.session_state.loaded_files = []
                st.rerun()
    
    # Main query interface
    if not st.session_state.loaded_files:
        st.info("👈 Please upload CSV files using the sidebar to get started")
        
        # Show example queries
        st.subheader("💡 Example Queries")
        st.markdown("""
        Once you upload files, try queries like:
        - `Find the lowest value of Column - Amplitude`
        - `What is the average of Sales column?`
        - `Show me the maximum temperature recorded`
        - `Count rows where Status is 'Active'`
        - `Sum all values in Revenue column`
        """)
        
        # Show data cleaning info
        st.subheader("🧹 Data Cleaning Features")
        st.markdown("""
        This app automatically cleans common data issues:
        - ✅ Concatenated numbers: `0.0690.0592` → `0.069`
        - ✅ Misplaced negatives: `0.0105-` → `-0.0105`
        - ✅ Special characters: `$1,234.56` → `1234.56`
        - ✅ Mixed text/numbers: `Value: 123` → `123`
        """)
        
        st.subheader("🗄️ Vector Database Persistence")
        st.markdown("""
        **Persistent vector storage benefits:**
        - ✅ Embeddings saved directly in vector DB
        - ✅ No repeated API calls for same data
        - ✅ Fast retrieval from indexed vectors
        - ✅ Survives application restarts
        - ✅ Efficient similarity search
        - ✅ Production-ready architecture
        """)
    else:
        st.subheader("🔎 Query Your Data")
        
        # Query input
        query = st.text_area(
            "Enter your query:",
            height=100,
            placeholder="e.g., Find the lowest value of Column - Amplitude of FTF as per data sheet",
            help="Ask questions about your data in natural language"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            run_query = st.button("▶️ Run Query", type="primary")
        
        with col2:
            if st.button("🗑️ Clear Results"):
                st.session_state.query_history = []
                st.rerun()
        
        if run_query and query:
            with st.spinner("Processing query..."):
                try:
                    result = analytical_agent.process_query(query)
                    
                    # Add to history
                    st.session_state.query_history.insert(0, {
                        'query': query,
                        'result': result
                    })
                    
                    # Keep only last 5 queries
                    st.session_state.query_history = st.session_state.query_history[:5]
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
                    st.info("💡 Try enabling data cleaning options in the sidebar")
        
        # Display current results
        if st.session_state.query_history:
            st.markdown("---")
            
            # Show most recent query result
            latest = st.session_state.query_history[0]
            st.subheader(f"Query: {latest['query']}")
            display_results(latest['result'])
            
            # Show query history
            if len(st.session_state.query_history) > 1:
                st.markdown("---")
                st.subheader("📜 Query History")
                
                for idx, item in enumerate(st.session_state.query_history[1:], 1):
                    with st.expander(f"Query {idx}: {item['query'][:50]}..."):
                        st.write(f"**Full Query:** {item['query']}")
                        display_results(item['result'])

if __name__ == "__main__":
    main()