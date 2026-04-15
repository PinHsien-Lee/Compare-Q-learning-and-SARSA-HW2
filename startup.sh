#!/bin/bash

# startup.sh: Optimize dev startup for OpenSpec workflow

set -e # Exit on error

echo "========================================"
echo "      🚀 DEVELOPMENT STARTUP"
echo "========================================"

# 1. Pull latest code
echo "🔍 Checking for updates from GitHub..."
if git pull origin main; then
    echo "✅ Code is up to date."
else
    echo "⚠️ Git pull failed. Please check your connection or remote settings."
fi

# 2. Read handover document
echo ""
echo "📝 CURRENT STATUS (00-handover.md):"
echo "----------------------------------------"
if [ -f "00-handover.md" ]; then
    cat "00-handover.md"
else
    echo "Creating missing handover document..."
    echo "# 00-Handover Document" > 00-handover.md
    echo "Status: New start." >> 00-handover.md
fi
echo "----------------------------------------"

# 3. Suggest next actions from tasks.md
echo ""
echo "💡 SUGGESTED NEXT ACTIONS:"
TASKS_FILE="openspec/changes/compare-q-learning-sarsa/03-tasks.md"
if [ -f "$TASKS_FILE" ]; then
    echo "Checking $TASKS_FILE..."
    # Find the first incomplete task
    NEXT_TASK=$(grep "- \[ \]" "$TASKS_FILE" | head -n 1)
    if [ -z "$NEXT_TASK" ]; then
        echo "🎉 All tasks are complete! You can run 'npm run dev:ending' to wrap up."
    else
        echo "👉 Next pending task: $NEXT_TASK"
    fi
else
    echo "⚠️ tasks.md not found at $TASKS_FILE"
fi

echo "========================================"
echo "Ready for development. Good luck!"
