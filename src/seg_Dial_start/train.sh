#!/bin/bash

# Base directory for the project
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Training commands
python train.py --dataset CLASS_all --save_model_name CLASS --batch_size 8 --accum 3 --epoch 20 

# Preprocessing commands
python data_preprocess_tax.py --save_name _taxonomy
python data_preprocess.py --save_name CLASS_all

# Inference commands (Note: Update --ckpt to point to your local model weights)
# Example CLASS inference:
python get_seg_result.py \
    --ckpt "$PROJECT_ROOT/models/CLASS_all/46-188" \
    --out_json "$PROJECT_ROOT/results/model_46-188_class.json" \
    --data_dir "$PROJECT_ROOT/data/Upchieve_CLASS/all"

# Example TalkMoves inference:
python get_seg_result.py \
    --ckpt "$PROJECT_ROOT/models/TalkMoves_all/94-1520" \
    --out_json "$PROJECT_ROOT/results/model_94-1520_talkmoves.json" \
    --data_dir "$PROJECT_ROOT/data/TalkMoves/all" \
    --gpu 0 --batch_size 4