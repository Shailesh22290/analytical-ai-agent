#!/usr/bin/env python3
"""
Command-line interface for the Analytical AI Agent
"""
import argparse
import json
import sys
from pathlib import Path

from config.settings import settings
from src.agents.ingestion import csv_ingestion
from src.agents.analytical_agent import analytical_agent


def print_result(result: dict, pretty: bool = True):
    """Print analysis result"""
    if pretty:
        print("\n" + "="*80)
        if "error" in result:
            print(f"❌ ERROR: {result['error']}")
            if "details" in result:
                print(f"Details: {result['details']}")
            if "supported_intents" in result:
                print(f"\nSupported intents:")
                for intent in result["supported_intents"]:
                    print(f"  - {intent}")
        else:
            print(f"📊 NARRATIVE:")
            print(f"{result.get('narrative', 'No narrative generated')}")
            
            print(f"\n📈 COMPUTED NUMBERS:")
            for key, value in result.get('numbers', {}).items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                elif isinstance(value, list) and len(value) > 5:
                    print(f"  {key}: [{value[0]}, {value[1]}, ..., {value[-1]}] ({len(value)} items)")
                else:
                    print(f"  {key}: {value}")
            
            print(f"\n📋 RESULT TABLE (first 5 rows):")
            result_table = result.get('result_table', [])
            if result_table:
                for i, row in enumerate(result_table[:5]):
                    print(f"  Row {i+1}: {row}")
                if len(result_table) > 5:
                    print(f"  ... and {len(result_table) - 5} more rows")
            else:
                print("  (empty)")
        print("="*80 + "\n")
    else:
        print(json.dumps(result, indent=2))


def ingest_command(args):
    """Handle ingest command"""
    print(f"\n🔄 Ingesting CSV file: {args.file}")
    
    try:
        file_id, metadata = csv_ingestion.ingest_csv(
            args.file,
            file_id=args.file_id,
            vectorize=args.vectorize
        )
        
        print(f"\n✅ Successfully ingested file!")
        print(f"   File ID: {file_id}")
        print(f"   Rows: {metadata.num_rows}")
        print(f"   Columns: {metadata.num_columns}")
        print(f"   Numeric columns: {', '.join(metadata.numeric_columns)}")
        print(f"   Text columns: {', '.join(metadata.text_columns)}")
        
        if args.vectorize:
            print(f"   ✓ Embeddings created and stored")
        
    except Exception as e:
        print(f"\n❌ Error ingesting file: {e}")
        sys.exit(1)


def query_command(args):
    """Handle query command"""
    print(f"\n🤔 Processing query: {args.query}")
    
    try:
        result = analytical_agent.process_query(
            args.query,
            enhance_prompt=args.enhance
        )
        
        print_result(result, pretty=not args.json)
        
        # Optionally save to file
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✅ Result saved to {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error processing query: {e}")
        sys.exit(1)


def status_command(args):
    """Handle status command"""
    print("\n📊 Agent Status:")
    
    try:
        status = analytical_agent.get_status()
        
        print(f"   Status: {status['status']}")
        print(f"   Loaded files: {status['loaded_files']}")
        
        if status['files']:
            print(f"\n   Files:")
            for file_info in status['files']:
                print(f"     - {file_info['file_id']}: {file_info['filename']}")
                print(f"       Rows: {file_info['rows']}, Columns: {file_info['columns']}")
                print(f"       Numeric columns: {', '.join(file_info['numeric_columns'])}")
        
        print(f"\n   Supported intents:")
        for intent in status['supported_intents']:
            print(f"     - {intent}")
        
    except Exception as e:
        print(f"\n❌ Error getting status: {e}")
        sys.exit(1)


def interactive_mode():
    """Run in interactive mode"""
    print("\n" + "="*80)
    print("🤖 Analytical AI Agent - Interactive Mode")
    print("="*80)
    print("\nCommands:")
    print("  query <your question>  - Ask a question")
    print("  status                 - Show agent status")
    print("  help                   - Show this help")
    print("  quit/exit              - Exit interactive mode")
    print("\nType your command or question:")
    print("="*80 + "\n")
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\nCommands:")
                print("  query <your question>  - Ask a question")
                print("  status                 - Show agent status")
                print("  help                   - Show this help")
                print("  quit/exit              - Exit interactive mode\n")
                continue
            
            if user_input.lower() == 'status':
                status = analytical_agent.get_status()
                print(f"\n📊 Loaded files: {status['loaded_files']}")
                for file_info in status['files']:
                    print(f"   - {file_info['file_id']}: {file_info['rows']} rows")
                print()
                continue
            
            # Treat as query
            if user_input.lower().startswith('query '):
                user_input = user_input[6:].strip()
            
            result = analytical_agent.process_query(user_input)
            print_result(result, pretty=True)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Analytical AI Agent - Process CSV data with natural language"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest a CSV file')
    ingest_parser.add_argument('file', help='Path to CSV file')
    ingest_parser.add_argument('--file-id', help='Custom file ID')
    ingest_parser.add_argument('--no-vectorize', dest='vectorize', action='store_false',
                              help='Skip vectorization')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the agent')
    query_parser.add_argument('query', help='Natural language query')
    query_parser.add_argument('--enhance', action='store_true',
                             help='Enhance query for clarity')
    query_parser.add_argument('--json', action='store_true',
                             help='Output raw JSON')
    query_parser.add_argument('--output', '-o', help='Save result to file')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show agent status')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Validate settings
    try:
        settings.validate()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nPlease ensure:")
        print("  1. You have created a .env file")
        print("  2. GEMINI_API_KEY is set in the .env file")
        sys.exit(1)
    
    # Route to appropriate command
    if args.command == 'ingest':
        ingest_command(args)
    elif args.command == 'query':
        query_command(args)
    elif args.command == 'status':
        status_command(args)
    elif args.command == 'interactive':
        interactive_mode()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()