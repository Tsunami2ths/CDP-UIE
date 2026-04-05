#!/bin/bash
# ==============================================================================
# 🚀 CDP-UIE: Cross-Domain Progressive Underwater Image Enhancement
# 🧪 Test Stage 2: Local Detail Reconstruction (LDR) Network
# ==============================================================================

# 请将此处替换为你的数据集绝对路径
DATAROOT="/the_abs_path_of_data"

echo "Evaluating Stage 2 (Final Enhanced Output)..."

python test.py \
    --dataroot ${DATAROOT} \
    --model Stage2

echo "Testing finished! Check the results folder."