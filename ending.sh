#!/bin/bash

# ending.sh: Wrap up development session

echo "--- Ending Script ---"

# 1. Update tasks and check status
echo "Checking project status..."
if command -v openspec &> /dev/null; then
    openspec status
    
    # Check if complete (this is a simplified check)
    STATUS=$(openspec status --json | grep '"isComplete": true')
    if [ ! -z "$STATUS" ]; then
        echo "Project complete! Archiving change..."
        openspec archive
    fi
fi

# 2. Write handover document
echo "Writing handover document..."
echo "Enter a summary of work completed (Ctrl+D to finish):"
SUMMARY=$(cat)

cat <<EOF > handover.md
# Handover Document
**Date:** $(date)

## Summary of Work
$SUMMARY

## Next Steps
- Continue implementing tasks in tasks.md
- Run experiments and analyze results
EOF

echo "Handover document updated."

# 3. Push code to GitHub
echo "Pushing code to GitHub..."
git add .
git commit -m "Update development progress: $(date)"
git push origin main

echo "--- Goodbye! ---"
