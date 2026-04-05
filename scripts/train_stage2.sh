#!/bin/bash
# ==============================================================================
# 🚀 CDP-UIE: Cross-Domain Progressive Underwater Image Enhancement
# 🌊 Train Stage 2: Local Detail Reconstruction (LDR) Network
# ⚠️ Notice: You MUST finish training Stage 1 before running this script.
# ==============================================================================

# 请将此处替换为你的数据集绝对路径
DATAROOT="/the_abs_path_of_data"

# 第二阶段的基础学习率
LR=0.001

echo "Starting training for Stage 2 (Local Detail Reconstruction)..."
echo "Loading pre-trained Stage 1 weights automatically..."

python train.py \
    --dataroot ${DATAROOT} \
    --model Stage2 \
    --lr ${LR}

echo "Stage 2 training finished!"