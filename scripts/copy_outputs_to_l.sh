#!/bin/bash

set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-meyjohann1@juwels-booster.fz-juelich.de}
REMOTE_BASE=${REMOTE_BASE:-/p/project1/hai_1074/meyjohann1/thesis}
CONTROL_SOCKET=${CONTROL_SOCKET:-~/.ssh/juwels-master}

# Start SSH master connection to reuse authentication
ssh -M -S "${CONTROL_SOCKET}" -f -N "${REMOTE_HOST}"

# Use the master connection for all rsync commands
rsync -avz --progress -e "ssh -S ${CONTROL_SOCKET}" "${REMOTE_HOST}:${REMOTE_BASE}/wandb/" ./wandb/
rsync -avz --progress -e "ssh -S ${CONTROL_SOCKET}" "${REMOTE_HOST}:${REMOTE_BASE}/logs/" ./logs/

# For policy evaluation runs, sync FastSAC checkpoints as well.
rsync -avz --progress -e "ssh -S ${CONTROL_SOCKET}" "${REMOTE_HOST}:${REMOTE_BASE}/models/fast_sac/" ./models/fast_sac/

# Close the master connection
ssh -S "${CONTROL_SOCKET}" -O exit "${REMOTE_HOST}"

echo "Fixing broken symlinks in wandb files..."

# Fix broken symlinks in wandb files by replacing them with actual files
find ./wandb/*/files/ -type l \( -name "*.pt" -o -name "*.diff" \) | while read -r symlink; do
    echo "Fixing broken symlink: $symlink"
    
    # Get the target path (what the symlink points to)
    target=$(readlink "$symlink")
    
    # Extract the relative path from cluster path to local path
    if [[ $target == /p/project1/hai_1074/meyjohann1/thesis/* ]]; then
        # Convert cluster path to local path
        local_target="${target#/p/project1/hai_1074/meyjohann1/thesis/}"
        local_file="./$local_target"
        
        if [[ -f "$local_file" ]]; then
            echo "  Replacing with: $local_file"
            rm "$symlink"
            cp "$local_file" "$symlink"
        else
            echo "  Warning: Local file not found: $local_file"
        fi
    fi
done

echo "Symlink fixing complete."
