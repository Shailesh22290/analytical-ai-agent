"""
Example usage of the Analytical AI Agent
This demonstrates the complete workflow
"""
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.agents.ingestion import csv_ingestion
from src.agents.analytical_agent import analytical_agent


def create_sample_data():
    """Create sample CSV files for demonstration"""
    print("📝 Creating sample data...")
    
    # Create data directory
    data_dir = Path("data/input")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample 1: Sales data
    sales_data = {
        'product_id': range(1, 21),
        'product_name': [
            'Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Webcam',
            'Headphones', 'USB Cable', 'Desk Lamp', 'Chair', 'Desk',
            'Tablet', 'Stylus', 'Phone', 'Charger', 'Case',
            'Speaker', 'Microphone', 'Adapter', 'Stand', 'Backpack'
        ],
        'price': [
            899.99, 25.50, 75.00, 299.99, 89.99,
            149.99, 12.99, 45.00, 249.99, 399.99,
            599.99, 99.99, 799.99, 29.99, 39.99,
            199.99, 129.99, 19.99, 49.99, 79.99
        ],
        'units_sold': [
            45, 230, 150, 78, 95,
            120, 340, 85, 62, 48,
            72, 110, 88, 280, 190,
            65, 58, 210, 95, 125
        ],
        'category': [
            'Electronics', 'Accessories', 'Accessories', 'Electronics', 'Electronics',
            'Electronics', 'Accessories', 'Furniture', 'Furniture', 'Furniture',
            'Electronics', 'Accessories', 'Electronics', 'Accessories', 'Accessories',
            'Electronics', 'Electronics', 'Accessories', 'Accessories', 'Accessories'
        ],
        'rating': [
            4.5, 4.2, 4.3, 4.7, 4.1,
            4.6, 3.9, 4.4, 4.8, 4.5,
            4.4, 4.3, 4.6, 4.0, 4.2,
            4.5, 4.4, 4.1, 4.3, 4.6
        ]
    }
    
    df_sales = pd.DataFrame(sales_data)
    df_sales['revenue'] = df_sales['price'] * df_sales['units_sold']
    sales_path = data_dir / "sales_q1.csv"
    df_sales.to_csv(sales_path, index=False)
    
    # Sample 2: Sales data Q2 (for comparison)
    sales_data_q2 = sales_data.copy()
    sales_data_q2['units_sold'] = [
        52, 245, 165, 82, 102,
        135, 360, 92, 68, 51,
        79, 118, 95, 295, 205,
        70, 63, 225, 102, 135
    ]
    
    df_sales_q2 = pd.DataFrame(sales_data_q2)
    df_sales_q2['revenue'] = df_sales_q2['price'] * df_sales_q2['units_sold']
    sales_q2_path = data_dir / "sales_q2.csv"
    df_sales_q2.to_csv(sales_q2_path, index=False)
    
    print(f"✓ Created {sales_path}")
    print(f"✓ Created {sales_q2_path}")
    
    return str(sales_path), str(sales_q2_path)


def demo_basic_queries(file_id1, file_id2):
    """Demonstrate basic analytical queries"""
    print("\n" + "="*80)
    print("🔍 DEMO: Basic Analytical Queries")
    print("="*80)
    
    queries = [
        "What are the top 5 products by revenue?",
        "Show me all products with price greater than 200",
        "What's the average rating for products?",
        "Compare the average units sold between Q1 and Q2",
        "What are the top 3 best selling products in the Electronics category?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'─'*80}")
        print(f"Query {i}: {query}")
        print(f"{'─'*80}")
        
        result = analytical_agent.process_query(query)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            if 'details' in result:
                print(f"   Details: {result['details']}")
        else:
            print(f"\n📊 Narrative:")
            print(f"   {result['narrative']}")
            
            print(f"\n📈 Key Numbers:")
            numbers = result['numbers']
            for key, value in list(numbers.items())[:5]:  # Show first 5
                if isinstance(value, float):
                    print(f"   • {key}: {value:.2f}")
                elif isinstance(value, list) and len(value) <= 3:
                    print(f"   • {key}: {value}")
                elif isinstance(value, list):
                    print(f"   • {key}: [{value[0]}, ..., {value[-1]}] ({len(value)} items)")
                else:
                    print(f"   • {key}: {value}")
            
            print(f"\n📋 Sample Results:")
            result_table = result['result_table']
            for row in result_table[:3]:
                print(f"   {row}")
            if len(result_table) > 3:
                print(f"   ... and {len(result_table) - 3} more rows")


def demo_comparison_queries(file_id1, file_id2):
    """Demonstrate comparison queries across files"""
    print("\n" + "="*80)
    print("🔍 DEMO: Comparison Queries")
    print("="*80)
    
    # Direct pandas comparison (bypassing LLM for demo reliability)
    from src.utils.models import CompareAveragesParams, CompareTopParams
    from src.agents.pandas_engine import pandas_engine
    
    print("\n1. Compare average units sold Q1 vs Q2:")
    params = CompareAveragesParams(
        column="units_sold",
        file1_id=file_id1,
        file2_id=file_id2
    )
    result_table, numbers = pandas_engine.compare_averages(params)
    
    print(f"   Q1 Average: {numbers[f'{file_id1}_average']:.2f}")
    print(f"   Q2 Average: {numbers[f'{file_id2}_average']:.2f}")
    print(f"   Difference: {numbers['difference']:.2f} ({numbers['percent_difference']:.2f}%)")
    
    print("\n2. Compare top 5 revenue products Q1 vs Q2:")
    params = CompareTopParams(
        column="revenue",
        n=5,
        file1_id=file_id1,
        file2_id=file_id2
    )
    result_table, numbers = pandas_engine.compare_top(params)
    
    print(f"   Top 5 Q1 average revenue: ${numbers[f'{file_id1}_average']:.2f}")
    print(f"   Top 5 Q2 average revenue: ${numbers[f'{file_id2}_average']:.2f}")


def demo_filtering_and_sorting(file_id1):
    """Demonstrate filtering and sorting"""
    print("\n" + "="*80)
    print("🔍 DEMO: Filtering and Sorting")
    print("="*80)
    
    from src.utils.models import FilterThresholdParams, SortParams
    from src.agents.pandas_engine import pandas_engine
    
    print("\n1. Filter products with rating >= 4.5:")
    params = FilterThresholdParams(
        column="rating",
        operator=">=",
        value=4.5,
        file_id=file_id1
    )
    result_table, numbers = pandas_engine.filter_threshold(params)
    
    print(f"   Found {numbers['filtered_rows']} products out of {numbers['total_rows']}")
    print(f"   ({numbers['percentage_matched']:.1f}% of total)")
    for row in result_table[:3]:
        print(f"   • {row['product_name']}: {row['rating']} ⭐")
    
    print("\n2. Sort products by revenue (descending):")
    params = SortParams(
        column="revenue",
        ascending=False,
        file_id=file_id1,
        limit=5
    )
    result_table, numbers = pandas_engine.sort_data(params)
    
    print(f"   Top revenue: ${numbers['first_value']:.2f}")
    for row in result_table:
        print(f"   • {row['product_name']}: ${row['revenue']:.2f}")


def main():
    """Main demonstration"""
    print("\n" + "="*80)
    print("🤖 Analytical AI Agent - Complete Demonstration")
    print("="*80)
    
    # Validate configuration
    try:
        settings.validate()
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease ensure:")
        print("  1. You have a .env file in the project root")
        print("  2. GEMINI_API_KEY is set in the .env file")
        return
    
    # Create sample data
    sales_q1_path, sales_q2_path = create_sample_data()
    
    # Ingest data
    print("\n📥 Ingesting data...")
    file_id1, meta1 = csv_ingestion.ingest_csv(
        sales_q1_path,
        file_id="sales_q1",
        vectorize=True  # Enable for semantic search
    )
    print(f"   ✓ Loaded Q1: {meta1.num_rows} rows, {len(meta1.numeric_columns)} numeric columns")
    
    file_id2, meta2 = csv_ingestion.ingest_csv(
        sales_q2_path,
        file_id="sales_q2",
        vectorize=True
    )
    print(f"   ✓ Loaded Q2: {meta2.num_rows} rows, {len(meta2.numeric_columns)} numeric columns")
    
    # Show agent status
    status = analytical_agent.get_status()
    print(f"\n📊 Agent Status: {status['status']}")
    print(f"   Loaded files: {status['loaded_files']}")
    
    # Run demos
    demo_filtering_and_sorting(file_id1)
    demo_comparison_queries(file_id1, file_id2)
    
    # Try natural language queries (requires API key)
    print("\n" + "="*80)
    print("🤖 Testing Natural Language Queries (requires Gemini API)")
    print("="*80)
    
    try:
        result = analytical_agent.process_query(
            "What are the top 3 products by revenue in Q1?"
        )
        
        if "error" not in result:
            print(f"\n✅ LLM Integration Working!")
            print(f"   Narrative: {result['narrative'][:150]}...")
        else:
            print(f"\n⚠️  LLM returned: {result['error']}")
    except Exception as e:
        print(f"\n⚠️  Could not test LLM queries: {e}")
        print("   (This is normal if API key is not configured)")
    
    print("\n" + "="*80)
    print("✅ Demonstration Complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Run: python main.py interactive")
    print("  2. Try queries like:")
    print("     - 'What are the top 5 products by price?'")
    print("     - 'Compare average revenue between Q1 and Q2'")
    print("     - 'Show products with rating above 4.5'")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()