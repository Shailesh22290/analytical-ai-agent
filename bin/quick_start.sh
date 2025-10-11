#!/bin/bash

# Quickstart script for Analytical AI Agent
# Run this after setting up the project structure

set -e  # Exit on error

echo "🚀 Analytical AI Agent - Quickstart Setup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Step 1: Check Python
echo "✓ Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "  Found Python $PYTHON_VERSION"

# Step 2: Create virtual environment
echo ""
echo "✓ Creating virtual environment..."
if [ -d "venv" ]; then
    echo "  Virtual environment already exists"
else
    python3 -m venv venv
    echo "  Created venv/"
fi

# Step 3: Activate virtual environment
echo ""
echo "✓ Activating virtual environment..."
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null || {
    echo "⚠️  Could not auto-activate. Please run:"
    echo "    source venv/bin/activate  # On Linux/Mac"
    echo "    venv\\Scripts\\activate     # On Windows"
    exit 1
}
echo "  Virtual environment activated"

# Step 4: Upgrade pip
echo ""
echo "✓ Upgrading pip..."
pip install --upgrade pip -q
echo "  pip upgraded"

# Step 5: Install dependencies
echo ""
echo "✓ Installing dependencies..."
pip install -r requirements.txt -q
echo "  All dependencies installed"

# Step 6: Setup .env file
echo ""
echo "✓ Setting up environment..."
if [ -f ".env" ]; then
    echo "  .env file already exists"
else
    cp .env.example .env
    echo "  Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your GEMINI_API_KEY"
    echo ""
    read -p "Do you have a Gemini API key? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your Gemini API key: " API_KEY
        sed -i.bak "s/your_gemini_api_key_here/$API_KEY/" .env
        rm .env.bak 2>/dev/null || true
        echo "  ✓ API key configured"
    else
        echo ""
        echo "  Get your API key from: https://makersuite.google.com/app/apikey"
        echo "  Then edit .env file and add: GEMINI_API_KEY=your_key_here"
        echo ""
    fi
fi

# Step 7: Create sample data
echo ""
echo "✓ Creating sample data..."
python examples/example_usage.py --setup-only 2>/dev/null || {
    # If example_usage doesn't have --setup-only flag, create data manually
    mkdir -p data/input
    echo "  Sample data directory created"
}

# Step 8: Run tests
echo ""
echo "✓ Running tests..."
pytest tests/ -v -x 2>&1 | head -n 20 || {
    echo "  ⚠️  Some tests failed (this is OK if API key not configured)"
}

# Step 9: Final instructions
echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Make sure your GEMINI_API_KEY is configured in .env"
echo ""
echo "2. Run the example demonstration:"
echo "   python examples/example_usage.py"
echo ""
echo "3. Try the CLI:"
echo "   python main.py status"
echo "   python main.py ingest data/input/yourfile.csv"
echo "   python main.py query 'What are the top 5 items by price?'"
echo ""
echo "4. Use interactive mode:"
echo "   python main.py interactive"
echo ""
echo "=========================================="
echo ""
echo "📚 Full documentation: INSTALLATION.md"
echo "🐛 Issues? Check the troubleshooting section"
echo ""