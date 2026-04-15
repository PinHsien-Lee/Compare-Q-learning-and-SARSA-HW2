#!/bin/bash

# startup.sh: Prepare development environment

echo "--- Startup Script ---"

# 1. Pull latest code from GitHub
echo "Pulling latest code from GitHub..."
git pull origin main

# 2. Read handover document if it exists
HANDOVER_FILE="handover.md"
if [ -f "$HANDOVER_FILE" ]; then
    echo "Reading handover document..."
    cat "$HANDOVER_FILE"
else
    echo "No handover document found."
fi

# 3. Suggest next actions
echo ""
echo "--- Next Actions ---"
echo "Based on current status:"
# Check openspec status if possible
if command -v openspec &> /dev/null; then
    openspec status
else
    echo "Check tasks.md for pending work."
fi

echo "Happy coding!"
