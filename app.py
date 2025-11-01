#!/usr/bin/env python3
"""
Analytical AI Agent - Streamlit Multi-CSV + Document Query Interface
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
from src.agents.document_ingestion import document_ingestion
from src.agents.analytical_agent import analytical_agent

# Page config
st.set_page_config(
    page_title="Analytical AI Agent",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'loaded_files' not in st.session_state:
    st.session_state.loaded_files = []
if 'loaded_documents' not in st.session_state:
    st.session_state.loaded_documents = []
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
        
        size_bytes = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
        size_mb = size_bytes / 1024 / 1024
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
    
    if value_str.endswith('-'):
        value_str = '-' + value_str[:-1]
    
    value_str = re.sub(r'[^\d.\-eE+]', '', value_str)
    
    if value_str.count('.') > 1 or (value_str.count('-') > 1 and not value_str.startswith('-')):
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
        if df_cleaned[col].dtype == 'object':
            sample = df_cleaned[col].dropna().head(10)
            if len(sample) > 0:
                numeric_pattern = any(bool(re.search(r'\d', str(val))) for val in sample)
                
                if numeric_pattern:
                    cleaned_col = df_cleaned[col].apply(clean_numeric_string)
                    valid_ratio = cleaned_col.notna().sum() / len(cleaned_col)
                    if valid_ratio > 0.5:
                        df_cleaned[col] = cleaned_col
    
    return df_cleaned

def load_csv_file(uploaded_file, clean_data=True, use_vector_db=True):
    """Load a CSV file and ingest it with vector DB persistence"""
    try:
        df = pd.read_csv(uploaded_file)
        original_shape = df.shape
        
        if clean_data:
            df = clean_csv_data(df)
        
        file_hash = get_file_hash(df)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w') as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        file_id, metadata = csv_ingestion.ingest_csv(tmp_path, vectorize=True)
        
        os.unlink(tmp_path)
        
        return {
            'name': uploaded_file.name,
            'file_id': file_id,
            'file_hash': file_hash,
            'metadata': metadata,
            'status': 'success',
            'original_shape': original_shape,
            'cleaned_shape': df.shape,
            'cleaned': clean_data,
            'vector_db': use_vector_db,
            'type': 'csv'
        }
    except Exception as e:
        return {
            'name': uploaded_file.name,
            'status': 'error',
            'error': str(e),
            'type': 'csv'
        }

def load_document_file(uploaded_file):
    """Load a document file (TXT/DOCX) and ingest it"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix, mode='wb') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        file_id, metadata = document_ingestion.ingest_document(tmp_path, vectorize=True)
        
        os.unlink(tmp_path)
        
        return {
            'name': uploaded_file.name,
            'file_id': file_id,
            'metadata': metadata,
            'status': 'success',
            'type': 'document'
        }
    except Exception as e:
        return {
            'name': uploaded_file.name,
            'status': 'error',
            'error': str(e),
            'type': 'document'
        }

def display_results(result):
    """Display query results in a formatted way"""
    if 'error' in result:
        st.error(f"❌ Error: {result.get('details', result['error'])}")
        
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
    
    metadata = result.get('metadata', {})
    intent = metadata.get('intent', '')
    
    # Handle analysis queries (document_query)
    if intent == 'document_query':
        st.subheader("📊 Analysis Response")
        narrative = result.get('narrative', 'No response generated.')
        st.markdown(narrative)
        
        # Show retrieved chunks if available
        if 'result_table' in result and result['result_table']:
            with st.expander("📑 Retrieved Analysis Sections", expanded=False):
                for idx, chunk_data in enumerate(result['result_table'][:3], 1):
                    st.markdown(f"**Section {idx}** (Relevance: {chunk_data.get('similarity_score', 'N/A')})")
                    
                    if chunk_data.get('question'):
                        st.info(f"**Q:** {chunk_data['question']}")
                        st.success(f"**A:** {chunk_data['answer']}")
                        if chunk_data.get('analysis'):
                            st.warning(f"**Analysis:** {chunk_data['analysis']}")
                    else:
                        st.text(chunk_data.get('content', '')[:500])
                    st.markdown("---")
        
        # Show stats
        numbers = result.get('numbers', {})
        if numbers:
            with st.expander("📊 Query Statistics", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    if 'num_results' in numbers:
                        st.metric("Results Found", numbers['num_results'])
                with col2:
                    if 'avg_similarity' in numbers:
                        st.metric("Avg Relevance", f"{numbers['avg_similarity']:.3f}")
        return
    
    # Handle general queries
    if intent == 'general_query':
        st.subheader("💬 Answer")
        narrative = result.get('narrative', 'No response generated.')
        st.markdown(narrative)
        
        numbers = result.get('numbers', {})
        if numbers and any(k in numbers for k in ['csv_files', 'document_files', 'total_rows']):
            with st.expander("📊 Data Summary"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'csv_files' in numbers:
                        st.metric("CSV Files", numbers['csv_files'])
                with col2:
                    if 'document_files' in numbers:
                        st.metric("Documents", numbers['document_files'])
                with col3:
                    if 'total_rows' in numbers:
                        st.metric("Total Rows", numbers['total_rows'])
        return
    
    # Handle analytical queries (existing code)
    numbers = result.get('numbers', {})
    
    if 'narrative' in result and result['narrative']:
        st.info(f"💡 {result['narrative']}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Query Results")
        
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
        
        if 'result_table' in result and result['result_table']:
            st.subheader("📋 Result Table")
            result_df = pd.DataFrame(result['result_table'])
            st.dataframe(result_df, use_container_width=True)
            
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
        
        stat_mapping = {
            'average': ('Average', '📊'),
            'min': ('Minimum', '⬇️'),
            'max': ('Maximum', '⬆️'),
            'sum': ('Sum', '➕'),
            'count': ('Count', '🔢'),
            'filtered_rows': ('Filtered Rows', '🔍'),
            'total_rows': ('Total Rows', '📄')
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
    st.title("📊 Analytical AI Agent")
    st.markdown("Upload CSV files and documents to run analytical queries across your data")
    
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
                
                st.info("💡 Embeddings are persisted in the vector DB.")
                
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
                help="Fix numbers stuck together"
            )
            st.session_state.cleaning_options['convert_negative'] = st.checkbox(
                "Fix negative sign position",
                value=True,
                help="Convert '0.069-' to '-0.069'"
            )
            
            clean_enabled = any(st.session_state.cleaning_options.values())
        
        # File upload tabs
        tab1, tab2 = st.tabs(["📊 CSV Files", "📄 Documents"])
        
        with tab1:
            uploaded_csvs = st.file_uploader(
                "Upload CSV Files",
                type=['csv'],
                accept_multiple_files=True,
                help="Upload one or more CSV files"
            )
            
            if uploaded_csvs:
                if st.button("📤 Load CSV Files", type="primary"):
                    st.session_state.loaded_files = []
                    
                    with st.spinner("Loading CSV files..."):
                        progress_bar = st.progress(0)
                        
                        for idx, file in enumerate(uploaded_csvs):
                            file.seek(0)
                            result = load_csv_file(file, clean_data=clean_enabled)
                            st.session_state.loaded_files.append(result)
                            progress_bar.progress((idx + 1) / len(uploaded_csvs))
                    
                    success_count = sum(1 for f in st.session_state.loaded_files if f['status'] == 'success')
                    if success_count > 0:
                        st.success(f"✅ Loaded {success_count} CSV file(s)")
                    st.rerun()
        
        with tab2:
            uploaded_docs = st.file_uploader(
                "Upload Documents",
                type=['txt', 'docx'],
                accept_multiple_files=True,
                help="Upload TXT or DOCX files"
            )
            
            if uploaded_docs:
                if st.button("📤 Load Documents", type="primary"):
                    st.session_state.loaded_documents = []
                    
                    with st.spinner("Loading documents..."):
                        progress_bar = st.progress(0)
                        
                        for idx, file in enumerate(uploaded_docs):
                            file.seek(0)
                            result = load_document_file(file)
                            st.session_state.loaded_documents.append(result)
                            progress_bar.progress((idx + 1) / len(uploaded_docs))
                    
                    success_count = sum(1 for f in st.session_state.loaded_documents if f['status'] == 'success')
                    if success_count > 0:
                        st.success(f"✅ Loaded {success_count} document(s)")
                    st.rerun()
        
        # Display loaded CSV files
        if st.session_state.loaded_files:
            st.subheader("📂 Loaded CSV Files")
            for file_info in st.session_state.loaded_files:
                if file_info['status'] == 'success':
                    with st.expander(f"✅ {file_info['name']}"):
                        st.write(f"**Rows:** {file_info['metadata'].num_rows}")
                        st.write(f"**Columns:** {file_info['metadata'].num_columns}")
                        st.write(f"**File ID:** `{file_info['file_id']}`")
                        
                        if file_info.get('cleaned'):
                            st.info(f"🧹 Data cleaned")
                        
                        if file_info['metadata'].columns:
                            st.write("**Column Names:**")
                            for col in file_info['metadata'].columns[:5]:
                                st.text(f"  • {col}")
                            if len(file_info['metadata'].columns) > 5:
                                st.write(f"... and {len(file_info['metadata'].columns) - 5} more")
                else:
                    with st.expander(f"❌ {file_info['name']}", expanded=True):
                        st.error(f"**Error:** {file_info['error']}")
        
        # Display loaded documents
        if st.session_state.loaded_documents:
            st.subheader("📄 Loaded Documents")
            for doc_info in st.session_state.loaded_documents:
                if doc_info['status'] == 'success':
                    with st.expander(f"✅ {doc_info['name']}"):
                        st.write(f"**Type:** {doc_info['metadata'].document_type.upper()}")
                        st.write(f"**Characters:** {doc_info['metadata'].num_characters:,}")
                        st.write(f"**Chunks:** {doc_info['metadata'].num_chunks}")
                        st.write(f"**Q&A Pairs:** {doc_info['metadata'].num_qa_pairs}")
                        st.write(f"**File ID:** `{doc_info['file_id']}`")
                        
                        if doc_info['metadata'].has_questions:
                            st.success("📋 Contains Q&A pairs")
                else:
                    with st.expander(f"❌ {doc_info['name']}", expanded=True):
                        st.error(f"**Error:** {doc_info['error']}")
        
        # Clear all button
        if st.session_state.loaded_files or st.session_state.loaded_documents:
            if st.button("🗑️ Clear All Files"):
                st.session_state.loaded_files = []
                st.session_state.loaded_documents = []
                st.rerun()
    
    # Main query interface
    if not st.session_state.loaded_files and not st.session_state.loaded_documents:
        st.info("👈 Please upload CSV files or documents using the sidebar to get started")
        
        # Show example queries
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💡 CSV Query Examples")
            st.markdown("""
            - `Find the lowest value of Amplitude`
            - `What is the average of Sales column?`
            - `Show me the maximum temperature`
            - `Sort by price in descending order`
            - `Count rows where Status is Active`
            """)
        
        with col2:
            st.subheader("📄 Document Query Examples")
            st.markdown("""
            - `How much is the rise in envelope value?`
            - `Explain the analysis for kurtosis`
            - `What does the document say about harmonic energy?`
            - `Show Q1 from the document`
            - `Compare current vs best performance`
            """)
        
        st.subheader("🗄️ Vector Database Persistence")
        st.markdown("""
        **Benefits:**
        - ✅ Embeddings saved in vector DB
        - ✅ Fast semantic search
        - ✅ No repeated API calls
        - ✅ Production-ready architecture
        """)
    else:
        st.subheader("🔎 Query Your Data")
        
        # Query input
        query = st.text_area(
            "Enter your query:",
            height=100,
            placeholder="e.g., How much is the rise in current envelope's absolute value? Or: Find the lowest amplitude value",
            help="Ask questions about your CSV data or documents"
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
                    
                    st.session_state.query_history.insert(0, {
                        'query': query,
                        'result': result
                    })
                    
                    st.session_state.query_history = st.session_state.query_history[:5]
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
        
        # Display current results
        if st.session_state.query_history:
            st.markdown("---")
            
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