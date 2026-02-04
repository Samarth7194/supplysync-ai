#!/bin/bash
# Daily commit script for SupplySync AI
cd "$(dirname "$0")"

# Add all changes
git add .

# Commit with timestamp
git commit -m "Daily update: $(date '+%Y-%m-%d %H:%M')"

# Push to GitHub
git push

echo "Daily commit completed!"
