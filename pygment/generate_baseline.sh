#!/bin/bash

# Define the values for soft_update and expectile
step_delay=(0 1 2 3 4 5)

# Loop over the soft_update values
for soft_update in "${soft_updates[@]}"; do
    python PPO_train.py --step-delay $step_delay
done
