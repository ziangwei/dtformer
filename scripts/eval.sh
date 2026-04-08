#!/usr/bin/env bash
# DTFormer 评估启动脚本
# Usage:
#   bash scripts/eval.sh                                       # 默认单尺度
#   CHECKPOINT=checkpoints/.../best.pth bash scripts/eval.sh   # 指定权重
#   GPUS=2 MULTI_SCALE=1 bash scripts/eval.sh                  # 多尺度评估
#   SAVE_VIS=1 bash scripts/eval.sh                            # 保存预测可视化

set -euo pipefail

# ─── 可修改参数 ───────────────────────────────────────────────
GPUS=${GPUS:-2}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29018}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CONFIG=${CONFIG:-"configs/experiments/nyu_dtformer_s.yaml"}
CHECKPOINT=${CHECKPOINT:-"checkpoints/best.pth"}

MULTI_SCALE=${MULTI_SCALE:-0}   # 1 = 多尺度测试
SAVE_VIS=${SAVE_VIS:-0}         # 1 = 保存预测彩图
VIS_MAX=${VIS_MAX:-20}          # 最多保存多少张可视化
EXTRA_ARGS=${EXTRA_ARGS:-""}

# GPU 可见性
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS - 1)))
fi
export CUDA_VISIBLE_DEVICES

# ─── 构建命令 ─────────────────────────────────────────────────
CMD="torchrun \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --nproc_per_node=${GPUS} \
    --master_port=${PORT} \
    tools/eval.py \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT}"

# 多尺度
if [ "${MULTI_SCALE}" = "1" ]; then
    CMD="${CMD} --multi-scale"
fi

# 可视化
if [ "${SAVE_VIS}" = "1" ]; then
    CMD="${CMD} --save-vis --vis-max ${VIS_MAX}"
fi

# 透传额外参数（如 --no-amp 等）
if [ -n "${EXTRA_ARGS}" ]; then
    CMD="${CMD} ${EXTRA_ARGS}"
fi

echo "========================================"
echo "  DTFormer Evaluation"
echo "  GPUs:       ${GPUS}"
echo "  Config:     ${CONFIG}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  MultiScale: ${MULTI_SCALE}"
echo "  SaveVis:    ${SAVE_VIS}"
echo "  CUDA:       ${CUDA_VISIBLE_DEVICES}"
echo "========================================"

exec ${CMD}
