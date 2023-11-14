#!/bin/bash

# Define the values for soft_update and expectile
soft_updates=(1 0.1)
expectiles=(0.5 0.6 0.7 0.8 0.9)

# Loop over the soft_update values
for soft_update in "${soft_updates[@]}"; do
    # Loop over the expectile values
    for expectile in "${expectiles[@]}"; do
        # Run the command 5 times
        for _ in {1..5}; do
            python IQL_train.py --expectile $expectile --soft_update $soft_update
        done
    done
done
