# debug_agent.py
# Enhanced debugging to identify the exact issue in the agent

from src.agents.ingestion import csv_ingestion
from src.agents.analytical_agent import analytical_agent
import pandas as pd
import pprint

# --- Step 1: Ingest the data ---
print("🔄 Ingesting CSV file...")
file_id, metadata = csv_ingestion.ingest_csv(
    "data/input/ftf_low_energy_table_numeric.csv",
    vectorize=True
)
print(f"✅ File ingested with ID: {file_id}\n")

# --- Step 2: Verify data is actually numeric ---
print("🔍 Verifying data types in memory...")
try:
    # Try to access the data directly from the ingestion module
    from src.agents.ingestion.csv_ingestion import get_dataframe
    df = get_dataframe(file_id)
    
    print("\nDataFrame Info:")
    print(df.dtypes)
    print("\nColumn:", 'Amplitude of FTF as per data sheet')
    print("Type:", df['Amplitude of FTF as per data sheet'].dtype)
    print("Sample values:", df['Amplitude of FTF as per data sheet'].head().tolist())
    
    # Test the operation manually
    print("\n🧪 Testing sort operation manually:")
    try:
        sorted_df = df.nsmallest(1, 'Amplitude of FTF as per data sheet')
        print("✅ Manual sort works!")
        print(sorted_df)
    except Exception as e:
        print(f"❌ Manual sort failed: {e}")
        
except ImportError:
    print("⚠️  Cannot import get_dataframe - checking differently...")
    # Alternative: load directly
    df = pd.read_csv("data/input/ftf_low_energy_table_numeric.csv")
    print("\nDirect file load:")
    print(df.dtypes)

# --- Step 3: Test agent with additional error details ---
print("\n" + "="*70)
print("Testing agent with error tracing...")
print("="*70)

user_query = "Find the lowest value of Column - Amplitude of FTF as per data sheet"

# Try to get more detailed error information
try:
    # Monkey-patch to get more details if possible
    import traceback
    result = analytical_agent.process_query(user_query)
    
    if result:
        if 'error' in result:
            print(f"\n❌ Error Type: {result.get('error')}")
            print(f"Details: {result.get('details', 'No details available')}")
            
            # Try to get the full stack trace if available
            if 'traceback' in result:
                print("\nFull Traceback:")
                print(result['traceback'])
        else:
            print("✅ Success!")
            pprint.pprint(result)
            
except Exception as e:
    print(f"\n💥 Exception during agent execution:")
    print(traceback.format_exc())

# --- Step 4: Test alternative queries ---
print("\n" + "="*70)
print("Testing alternative query formulations...")
print("="*70)

alternative_queries = [
    "What is the minimum Amplitude of FTF as per data sheet?",
    "Show me the row with the smallest Amplitude of FTF as per data sheet",
    "Get minimum value from Amplitude of FTF as per data sheet column",
]

for i, query in enumerate(alternative_queries, 1):
    print(f"\n{i}. Query: {query}")
    result = analytical_agent.process_query(query)
    if result and 'error' not in result:
        print("   ✅ Success!")
        pprint.pprint(result)
    else:
        print(f"   ❌ Failed: {result.get('error', 'Unknown') if result else 'No result'}")