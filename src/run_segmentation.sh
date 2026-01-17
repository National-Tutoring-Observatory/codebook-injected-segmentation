#!/bin/bash

# Configuration for Cornell AI Gateway (Use environment variables for security)
# export AI_GATEWAY_KEY='your_key_here'
export AI_GATEWAY_BASE_URL='https://api.ai.it.cornell.edu'
export AI_GATEWAY_PROVIDER='google.gemini-3-pro-preview'

# Directory paths relative to the project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/TalkMoves/all"
OUT_DIR="$PROJECT_ROOT/data/llm_result"
OUT_JSON="$OUT_DIR/llm_results_geminipro3_talkmoves.json"

mkdir -p "$OUT_DIR"

echo "Running LLM Segmentation..."
echo "Data Directory: $DATA_DIR"
echo "Output JSON: $OUT_JSON"

python3 "$PROJECT_ROOT/src/seg_LLM/llm_segmentation_TalkMoves_tax.py" \
    --data_dir "$DATA_DIR" \
    --out_json "$OUT_JSON" \
   # --limit 10

echo "Done! Results saved to $OUT_JSON"

