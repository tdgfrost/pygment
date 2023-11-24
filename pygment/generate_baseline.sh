#!/bin/bash
# Define the values for soft_update and expectile
step_delays=(0 1 2 3 4 5)
# Loop over the soft_update values
for step_delay in "${step_delays[@]}"; do
    python PPO_train.py --step-delay $step_delay
done
