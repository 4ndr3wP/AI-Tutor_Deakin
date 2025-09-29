#!/bin/bash

# Get script directory and load config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

echo "🔄 Syncing to remote frontend server..."

# Sync the entire project directory to frontend server
rsync -avz --delete \
    --exclude='.git/' \
    --exclude='AI-Tutor/RL_db_reference_1k_500/*.sqlite3' \
    --exclude='AI-Tutor/RL_db_reference_1k_500/[0-9a-f]*-[0-9a-f]*/' \
    --exclude='node_modules/' \
    --exclude='__pycache__/' \
    "${SCRIPT_DIR}/" \
    $USERNAME@$FRONTEND_SERVER:/home/s222539911/AI-Tutor_Deakin/

echo "✅ Sync completed"
