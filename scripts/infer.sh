#!/usr/bin/env bash
# DTFormer 单图推理脚本
# Usage:
#   bash scripts/infer.sh                                                    # 默认参数
#   CHECKPOINT=checkpoints/.../best.pth RGB=demo/rgb.jpg DEPTH=demo/depth.png bash scripts/infer.sh
#   MULTI_SCALE=1 bash scripts/infer.sh                                      # 多尺度推理

set -euo pipefail

# ─── 可修改参数 ───────────────────────────────────────────────
CONFIG=${CONFIG:-"configs/experiments/nyu_dtformer_s.yaml"}
CHECKPOINT=${CHECKPOINT:-"checkpoints/best.pth"}

RGB=${RGB:-"demo/rgb.jpg"}
DEPTH=${DEPTH:-"demo/depth.png"}
OUTPUT=${OUTPUT:-"output/prediction.png"}

MULTI_SCALE=${MULTI_SCALE:-0}
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

if [ "${MULTI_SCALE}" = "1" ]; then
    CMD="${CMD} --multi-scale"
fi

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
