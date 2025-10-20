#!/bin/bash

# Start SSH master connection to reuse authentication
ssh -M -S ~/.ssh/juwels-master -f -N meyjohann1@juwels-booster.fz-juelich.de

# Use the master connection for all rsync commands
rsync -avz --progress -e "ssh -S ~/.ssh/juwels-master" meyjohann1@juwels-booster.fz-juelich.de:/p/home/jusers/meyjohann1/juwels/meyjohann1/thesis/wandb/ ./wandb/
# rsync -avz --progress -e "ssh -S ~/.ssh/juwels-master" meyjohann1@juwels-booster.fz-juelich.de:/p/home/jusers/meyjohann1/juwels/meyjohann1/thesis/models/ ./models/
rsync -avz --progress -e "ssh -S ~/.ssh/juwels-master" meyjohann1@juwels-booster.fz-juelich.de:/p/home/jusers/meyjohann1/juwels/meyjohann1/thesis/logs/ ./logs/

# Close the master connection
ssh -S ~/.ssh/juwels-master -O exit meyjohann1@juwels-booster.fz-juelich.de

echo "🔧 Fixing broken symlinks in wandb files..."

# Fix broken symlinks in wandb files by replacing them with actual files
find ./wandb/*/files/ -type l -name "*.pt" -o -name "*.diff" | while read -r symlink; do
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

echo "✅ Symlink fixing complete!"