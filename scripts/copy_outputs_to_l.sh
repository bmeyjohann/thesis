#!/bin/bash

# Start SSH master connection to reuse authentication
ssh -M -S ~/.ssh/juwels-master -f -N meyjohann1@juwels-booster.fz-juelich.de

# Use the master connection for all rsync commands
rsync -avz --progress -e "ssh -S ~/.ssh/juwels-master" meyjohann1@juwels-booster.fz-juelich.de:/p/home/jusers/meyjohann1/juwels/meyjohann1/thesis/models/ ./models/
rsync -avz --progress -e "ssh -S ~/.ssh/juwels-master" meyjohann1@juwels-booster.fz-juelich.de:/p/home/jusers/meyjohann1/juwels/meyjohann1/thesis/logs/ ./logs/
rsync -avz --progress -e "ssh -S ~/.ssh/juwels-master" meyjohann1@juwels-booster.fz-juelich.de:/p/home/jusers/meyjohann1/juwels/meyjohann1/thesis/wandb/ ./wandb/

# Close the master connection
ssh -S ~/.ssh/juwels-master -O exit meyjohann1@juwels-booster.fz-juelich.de