#!/bin/bash
# AI-Tutor Control Script - Configurable Version
set -e

# Load configuration
CONFIG_FILE="$(dirname "$0")/config.env"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    echo "📋 Please copy config.env.template to config.env and fill in your details:"
    echo "   cp config.env.template config.env"
    echo "   # Then edit config.env with your paths and credentials"
    exit 1
fi

# Source the configuration
source "$CONFIG_FILE"

# Validate required variables
required_vars=("BACKEND_DIR" "FRONTEND_DIR" "USERNAME" "FRONTEND_SERVER")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Required variable $var is not set in config.env"
        exit 1
    fi
done

log() { echo -e "\033[0;34m[$(date '+%H:%M:%S')]\033[0m $1"; }
success() { echo -e "\033[0;32m✅ $1\033[0m"; }
error() { echo -e "\033[0;31m❌ $1\033[0m"; }

# Cleanup function
cleanup_all() {
    log "🛑 Stopping all AI-Tutor services..."
    tmux kill-session -t ai-backend 2>/dev/null || true
    tmux kill-session -t ai-frontend 2>/dev/null || true
    pkill -f "rag_multi_query.py" 2>/dev/null || true
    pkill -f "npm run dev" 2>/dev/null || true
    lsof -ti:$BACKEND_PORT | xargs -r kill -9 2>/dev/null || true
    lsof -ti:$FRONTEND_PORT | xargs -r kill -9 2>/dev/null || true
    success "All services stopped"
}

# Start backend with configured paths
start_backend() {
    log "🚀 Starting backend in tmux..."
    
    tmux kill-session -t ai-backend 2>/dev/null || true
    pkill -f "rag_multi_query.py" 2>/dev/null || true
    lsof -ti:$BACKEND_PORT | xargs -r kill -9 2>/dev/null || true
    sleep 3
    
    tmux new-session -d -s ai-backend
    tmux send-keys -t ai-backend "cd $BACKEND_DIR" Enter
    tmux send-keys -t ai-backend "module load Anaconda3/2024.02" Enter
    tmux send-keys -t ai-backend "eval \"\$(conda shell.bash hook)\"" Enter
    tmux send-keys -t ai-backend "conda activate test_app" Enter
    tmux send-keys -t ai-backend "python rag_multi_query.py" Enter
    
    log "⏳ Waiting for backend (model loading may take minutes)..."
    for i in {1..120}; do
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:$BACKEND_PORT/health 2>/dev/null | grep -q "200\|404"; then
            success "Backend ready at http://localhost:$BACKEND_PORT"
            return 0
        fi
        [ $((i % 15)) -eq 0 ] && log "Still loading model... ($i/120)"
        sleep 2
    done
    error "Backend failed to start within 4 minutes"
    return 1
}

# Start frontend with configured paths and credentials
start_frontend() {
    log "🌐 Starting frontend in tmux..."
    tmux kill-session -t ai-frontend 2>/dev/null || true
    tmux new-session -d -s ai-frontend
    tmux send-keys -t ai-frontend "ssh $USERNAME@$FRONTEND_SERVER" Enter
    sleep 3
    tmux send-keys -t ai-frontend "cd $FRONTEND_DIR" Enter
    tmux send-keys -t ai-frontend "npm run dev -- --host" Enter
    success "Frontend starting at http://$FRONTEND_HOST:$FRONTEND_PORT"
}

# Restart function
restart_backend() {
    log "♻️ Restarting backend..."
    if tmux has-session -t ai-backend 2>/dev/null; then
        tmux send-keys -t ai-backend C-c
        sleep 5
        pkill -9 -f "rag_multi_query.py" 2>/dev/null || true
        tmux send-keys -t ai-backend "python rag_multi_query.py" Enter
        
        log "⏳ Waiting for backend restart..."
        for i in {1..90}; do
            if curl -s -o /dev/null -w "%{http_code}" http://localhost:$BACKEND_PORT/health 2>/dev/null | grep -q "200"; then
                success "Backend restarted successfully!"
                return 0
            fi
            [ $((i % 15)) -eq 0 ] && log "Still restarting... ($i/90)"
            sleep 2
        done
        error "Restart failed - try full backend start instead"
        return 1
    else
        log "No backend session found, starting fresh..."
        start_backend
    fi
}

# Status function
show_status() {
    echo ""
    echo "=== AI-Tutor Status ==="
    tmux list-sessions 2>/dev/null | grep ai-backend >/dev/null && success "Backend tmux: ✓" || error "Backend tmux: ✗"
    tmux list-sessions 2>/dev/null | grep ai-frontend >/dev/null && success "Frontend tmux: ✓" || error "Frontend tmux: ✗"
    lsof -ti:$BACKEND_PORT >/dev/null 2>&1 && success "Backend port: ✓" || error "Backend port: ✗"
    echo ""
    echo "🌐 Frontend: http://$FRONTEND_HOST:$FRONTEND_PORT"
    echo "🔧 Backend: http://localhost:$BACKEND_PORT"
    echo "📺 Monitor: tmux list-sessions"
}

# Command handling
case "${1:-help}" in
    "backend"|"b")
        cleanup_all && start_backend && show_status
        ;;
    "frontend"|"f")
        start_frontend && show_status
        ;;
    "full"|"all")
        log "🚀 Starting full stack..."
        cleanup_all && start_backend && sleep 5 && start_frontend
        success "Full stack ready!" && show_status
        ;;
    "stop"|"s")
        cleanup_all && show_status
        ;;
    "restart"|"r")
        restart_backend && show_status
        ;;
    "status")
        show_status
        ;;
    *)
        echo "🧠 AI-Tutor Control Script"
        echo "Commands: backend, frontend, full, stop, restart, status"
        echo "Examples: $0 full    # Start everything"
        ;;
esac