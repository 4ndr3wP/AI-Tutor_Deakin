#!/bin/bash

# Get script directory and load config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

# Load conda environment
source /sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh
conda activate test_app

echo "🔍 Starting file watcher..."
echo "Changes will be automatically synced to frontend server"

# Watch for changes and sync
watchmedo shell-command \
    --patterns="*.py;*.js;*.jsx;*.ts;*.tsx;*.css;*.html" \
    --recursive \
    --command="${SCRIPT_DIR}/auto_rsync.sh" \
    "${SCRIPT_DIR}/"
