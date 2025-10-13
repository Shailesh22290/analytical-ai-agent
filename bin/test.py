"""
Updated display_results function for app.py
Replace your existing display_results function with this
"""

def display_results(result):
    """Display query results in a formatted way"""
    print(f"[DEBUG] display_results called with keys: {result.keys()}")
    print(f"[DEBUG] narrative length: {len(result.get('narrative', ''))}")
    
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
    
    # Check if this is a general/descriptive query
    is_general_query = result.get('metadata', {}).get('intent') == 'general_query'
    narrative = result.get('narrative', '')
    numbers = result.get('numbers', {})
    
    print(f"[DEBUG] is_general_query: {is_general_query}")
    print(f"[DEBUG] has narrative: {len(narrative) > 0}")
    
    # Show narrative first (for both query types)
    if narrative:
        st.success("✅ Response")
        st.markdown(narrative)
    else:
        st.warning("⚠️ No narrative generated")
    
    # For general queries, show metadata in expander
    if is_general_query:
        with st.expander("ℹ️ Query Information"):
            st.json(numbers)
            if result.get('metadata'):
                st.write("**Query Details:**")
                st.json(result['metadata'])
        return  # Don't show result table/stats for general queries
    
    # For analytical queries, show detailed results
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
                st.dataframe(results_df, width='stretch')
            else:
                st.metric("Result Value", values)
        
        # Show result table if available
        if 'result_table' in result and result['result_table']:
            st.subheader("📋 Result Table")
            result_df = pd.DataFrame(result['result_table'])
            st.dataframe(result_df, width='stretch')
            
            # Add download button with unique key
            csv = result_df.to_csv(index=False)
            import uuid
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