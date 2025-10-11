#!/bin/bash
# Find the execution file that's creating the problematic structure

echo "🔍 Looking for execution files..."
echo ""

# Find files that might contain the top_n execution logic
echo "Files containing 'top_n' logic:"
grep -r "def.*top_n\|top_values\|top_indices" src/ --include="*.py" -l

echo ""
echo "Content of execution-related files:"
echo "===================================="

# Show content of likely execution files
for file in $(find src/ -name "*execution*.py" -o -name "*executor*.py" -o -name "*analytical*.py"); do
    if [ -f "$file" ]; then
        echo ""
        echo "📄 File: $file"
        echo "---"
        head -50 "$file"
        echo "..."
    fi
done