#!/bin/bash
# ==============================================================================
# 🚀 CDP-UIE: Cross-Domain Progressive Underwater Image Enhancement
# 🌊 Train Stage 1: Global Refinement (GR) Network
# ==============================================================================

# 请将此处替换为你的数据集绝对路径
DATAROOT="/the_abs_path_of_data"

# 第一阶段的基础学习率
LR=0.0005

echo "Starting training for Stage 1 (Global Refinement)..."

python train.py \
    --dataroot ${DATAROOT} \
    --model Stage1 \
    --lr ${LR}

echo "Stage 1 training finished!"