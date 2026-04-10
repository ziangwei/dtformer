#!/usr/bin/env bash
# DTFormer 单图推理脚本
# Usage:
#   bash scripts/infer.sh
#   CHECKPOINT=checkpoints/.../best.pth RGB=demo/rgb.jpg DEPTH=demo/depth.png bash scripts/infer.sh
#   EXTRA_ARGS="--text-mode image_specific --labels wall floor table" bash scripts/infer.sh

set -euo pipefail

# ─── 可修改参数 ───────────────────────────────────────────────
CONFIG=${CONFIG:-"configs/experiments/nyu_dtformer_s.yaml"}
CHECKPOINT=${CHECKPOINT:-"checkpoints/best.pth"}

RGB=${RGB:-"demo/rgb.jpg"}
DEPTH=${DEPTH:-"demo/depth.png"}
OUTPUT=${OUTPUT:-"output/prediction.png"}

EXTRA_ARGS=${EXTRA_ARGS:-""}

# 单卡即可
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ─── 构建命令 ─────────────────────────────────────────────────
CMD="python tools/infer.py \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --rgb ${RGB} \
    --depth ${DEPTH} \
    --output ${OUTPUT}"

if [ -n "${EXTRA_ARGS}" ]; then
    CMD="${CMD} ${EXTRA_ARGS}"
fi

echo "========================================"
echo "  DTFormer Inference"
echo "  Config:     ${CONFIG}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  RGB:        ${RGB}"
echo "  Depth:      ${DEPTH}"
echo "  Output:     ${OUTPUT}"
echo "========================================"

exec ${CMD}
