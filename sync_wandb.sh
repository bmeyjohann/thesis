#!/bin/bash
# Sync offline wandb runs to cloud from login node
# Usage: bash sync_wandb.sh

echo "🔄 Syncing offline wandb runs from JUWELS compute nodes..."

# Activate environment for wandb commands
cd sc_venv_template
source activate.sh
cd ..

# Check for offline runs in various locations
SYNC_COUNT=0

# Location 1: User home wandb directory
if [ -d "$HOME/.wandb" ]; then
    echo "📂 Checking $HOME/.wandb for offline runs..."
    find $HOME/.wandb -name "offline-run-*" -type d | while read run_dir; do
        echo "  Syncing: $(basename $run_dir)"
        wandb sync "$run_dir" && echo "    ✅ Success" || echo "    ❌ Failed"
        ((SYNC_COUNT++))
    done
fi

# Location 2: Working directory wandb folder
if [ -d "./wandb" ]; then
    echo "📂 Checking ./wandb for offline runs..."
    find ./wandb -name "offline-run-*" -type d | while read run_dir; do
        echo "  Syncing: $(basename $run_dir)"
        wandb sync "$run_dir" && echo "    ✅ Success" || echo "    ❌ Failed"
        ((SYNC_COUNT++))
    done
fi

# Location 3: Check recent log files for wandb dirs
echo "📂 Checking job logs for wandb run locations..."
if [ -d "./logs" ]; then
    grep -h "Logs saved locally to:" logs/*.out 2>/dev/null | while read -r line; do
        run_dir=$(echo "$line" | sed 's/.*Logs saved locally to: //')
        if [ -d "$run_dir" ]; then
            echo "  Syncing: $(basename $run_dir)"
            wandb sync "$run_dir" && echo "    ✅ Success" || echo "    ❌ Failed"
            ((SYNC_COUNT++))
        fi
    done
fi

echo ""
echo "🎉 Wandb sync completed!"
echo "📊 Check your dashboard at: https://wandb.ai/<username>/juwels-reward-pilot"
echo ""
echo "💡 Tip: You can also manually sync specific runs with:"
echo "   wandb sync /path/to/wandb/run-directory"
