#!/bin/bash

# Make sure you've updated your .env file with your actual wandb API key
# WANDB_API_KEY=your_actual_api_key_here

# Set configuration
CONFIG="small"

# Generate timestamp for unique run ID
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="${CONFIG}_run_${TIMESTAMP}"

# Run WT5 experiment with wandb logging enabled
python -m models.hf.WT5.run_experiment \
  --config ${CONFIG} \
  --output_dir "./models/hf/WT5/model_outputs/${RUN_NAME}" \
  --batch_size 8 \
  --epochs 3 \
  --learning_rate 5e-5 \
  --max_length 512 \
  --gradient_accumulation_steps 1 \
  --eval_steps 500 \
  --logging_steps 50 \
  --save_steps 1000 \
  --use_wandb \
  --wandb_project "wt5-sentiment" 