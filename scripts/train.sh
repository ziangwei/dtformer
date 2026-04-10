#!/usr/bin/env bash
# DTFormer 训练启动脚本
# Usage:
#   bash scripts/train.sh                              # 默认: NYU + DTFormer-S, 2 GPUs
#   GPUS=4 CONFIG=configs/experiments/nyu_dtformer_s.yaml bash scripts/train.sh
#   GPUS=1 CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh   # 单卡调试

set -euo pipefail

# ─── 可修改参数 ───────────────────────────────────────────────
GPUS=${GPUS:-2}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29119}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CONFIG=${CONFIG:-"configs/experiments/nyu_dtformer_s.yaml"}
RESUME=${RESUME:-""}          # 断点续训: RESUME=checkpoints/.../epoch-100.pth
EXTRA_ARGS=${EXTRA_ARGS:-""}  # 透传额外参数

# GPU 可见性（默认按 GPUS 数自动生成 0,1,...）
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS - 1)))
fi
export CUDA_VISIBLE_DEVICES

# HuggingFace 缓存（LRZ 集群专用，本地可注释掉）
# CACHE_DIR="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/huggingface_cache"
# mkdir -p "${CACHE_DIR}" && export HF_HOME="${CACHE_DIR}"

# ─── 构建命令 ─────────────────────────────────────────────────
CMD="torchrun \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --nproc_per_node=${GPUS} \
    --master_port=${PORT} \
    tools/train.py \
    --config ${CONFIG}"

# 断点续训
if [ -n "${RESUME}" ]; then
    CMD="${CMD} --resume ${RESUME}"
fi

# 透传额外参数（如 --torch-compile, --no-tensorboard, --no-amp 等）
if [ -n "${EXTRA_ARGS}" ]; then
    CMD="${CMD} ${EXTRA_ARGS}"
fi

echo "========================================"
echo "  DTFormer Training"
echo "  GPUs:   ${GPUS}"
echo "  Config: ${CONFIG}"
echo "  CUDA:   ${CUDA_VISIBLE_DEVICES}"
[ -n "${RESUME}" ] && echo "  Resume: ${RESUME}"
echo "========================================"

exec ${CMD}
