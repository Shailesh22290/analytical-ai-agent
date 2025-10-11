#!/usr/bin/env python3
"""
Analytical AI Agent - Simple Query Runner
Edit the CSV_FILE and QUERY variables below, then run: python run_query.py
"""
from src.agents.ingestion import csv_ingestion
from src.agents.analytical_agent import analytical_agent
import sys

# ============================================================================
# EDIT THESE TWO LINES FOR YOUR QUERY
# ============================================================================
CSV_FILE = "data/input/ftf_low_energy_table_numeric.csv"
QUERY = "What is the average of Amplitude of FTF as per data sheet?"
# ============================================================================


def main():
    try:
        # Load CSV
        print(f"Loading: {CSV_FILE}...")
        file_id, metadata = csv_ingestion.ingest_csv(CSV_FILE, vectorize=True)
        print(f"✓ Loaded {metadata.num_rows} rows\n")
        
        # Run query
        print(f"Query: {QUERY}\n")
        result = analytical_agent.process_query(QUERY)
        
        # Check for errors
        if 'error' in result:
            print(f"❌ Error: {result.get('details', result['error'])}")
            sys.exit(1)
        
        # Display results
        print("="*70)
        print("RESULTS")
        print("="*70)
        
        numbers = result.get('numbers', {})
        
        # Show values found
        if 'top_values' in numbers:
            values = numbers['top_values']
            indices = numbers.get('top_indices', [])
            
            if isinstance(values, list):
                for val, idx in zip(values, indices):
                    print(f"Value: {val} (Row Index: {idx})")
            else:
                print(f"Value: {values}")
        
        # Show statistics
        for key in ['average', 'min', 'max', 'sum', 'count', 'filtered_rows', 'total_rows']:
            if key in numbers:
                print(f"{key.replace('_', ' ').title()}: {numbers[key]}")
        
        # Show row count
        if 'result_table' in result:
            print(f"Rows Returned: {len(result['result_table'])}")
        
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()