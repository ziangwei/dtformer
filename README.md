# DTFormer

Tri-modal (RGB-Depth-Text) semantic segmentation framework. DTFormer introduces Temperature-Scaled Cosine Attention (TSCA) for text-guided feature alignment at both the encoder (TSA-E) and decoder (TSA-D) stages.

Built on the DFormerv2 backbone with all OpenMMLab (mmseg / mmcv) dependencies removed.

## Environment Setup

Tested with Python 3.10, PyTorch 2.1, CUDA 11.8.

```bash
# 1. Create conda environment
conda create -n dtformer python=3.10 -y
conda activate dtformer

# 2. Install PyTorch (match your CUDA version)
#    CUDA 11.8:
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
#    CUDA 12.1:
# conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. Install dependencies
pip install -r requirements.txt
```

No mmseg / mmcv / mmengine needed.

## Data Preparation

All datasets live under `datasets/` (gitignored). Create the directory and organize as follows:

```
datasets/
├── NYUDepthv2/
│   ├── RGB/                  # RGB images (.jpg)
│   ├── Depth/                # Depth maps (.png, 3-channel replicated)
│   ├── Label/                # Semantic labels (.png, 1-indexed, 0 = unlabeled)
│   ├── train.txt             # Training split (one filename stem per line)
│   ├── test.txt              # Test split
│   ├── nyu40_labels.txt      # 40 class names, one per line
│   ├── image_labels.json     # Per-image text labels (from VLM)
│   └── cache/
│       └── vocab_embeds.pt   # CLIP embeddings for full vocabulary
│
└── SUNRGBD/
    ├── RGB/                  # RGB images (.jpg)
    ├── Depth/                # Depth maps (.png, 3-channel replicated)
    ├── labels/               # Semantic labels (.png) — 注意小写
    ├── train.txt
    ├── test.txt
    ├── sunrgbd37_labels.txt  # 37 class names
    ├── image_labels.json
    └── cache/
        └── vocab_embeds.pt
```

NYUDepthv2: 795 train / 654 test, 40 classes, 480 x 640.
SUNRGBD: 5285 train / 5050 test, 37 classes, 480 x 480.

### Build CLIP Text Cache

Before training, pre-compute the CLIP text embeddings (only needed once per dataset):

```bash
# NYU vocabulary embeddings
python tools/build_clip_cache.py \
    --dataset NYUDepthv2 \
    --output datasets/NYUDepthv2/cache/vocab_embeds.pt

# SUNRGBD vocabulary embeddings
python tools/build_clip_cache.py \
    --dataset SUNRGBD \
    --output datasets/SUNRGBD/cache/vocab_embeds.pt
```

For `image_specific` text mode, the per-image label list (`image_labels.json`) is looked up at runtime against the vocabulary embedding table — no separate per-image embedding cache is needed.

### Generate Image Labels with VLM (Optional)

If you need to generate per-image text labels from scratch using a VLM:

```bash
# InternVL3
python tools/generate_tags_internvl.py \
    --dataset nyu \
    --dataset_dir datasets/NYUDepthv2 \
    --output_file image_labels_internvl.json

# Qwen3-VL
python tools/generate_tags_qwen.py \
    --dataset_dir datasets/NYUDepthv2 \
    --output_file image_labels_qwen.json

# Take intersection of multiple VLM outputs
python tools/merge_image_labels.py \
    datasets/NYUDepthv2/image_labels_internvl.json \
    datasets/NYUDepthv2/image_labels_qwen.json \
    --output datasets/NYUDepthv2/image_labels.json
```

## Training

```bash
# Default: NYU + DTFormer-S, 2 GPUs
bash scripts/train.sh

# Custom config and GPU count
GPUS=4 CONFIG=configs/experiments/nyu_dtformer_b.yaml bash scripts/train.sh

# Resume from checkpoint
RESUME=checkpoints/NYUDepthv2_DTFormer_S/epoch-300.pth bash scripts/train.sh

# Extra options (torch.compile, disable TensorBoard, etc.)
EXTRA_ARGS="--torch-compile --no-amp" bash scripts/train.sh
```

Or call `torchrun` directly:

```bash
torchrun --nproc_per_node=2 tools/train.py \
    --config configs/experiments/nyu_dtformer_s.yaml
```

TensorBoard logs are written to `<log_dir>/tb/`:

```bash
tensorboard --logdir checkpoints/NYUDepthv2_DTFormer_S/tb
```

## Evaluation

```bash
# Single-scale evaluation
CHECKPOINT=checkpoints/.../best.pth bash scripts/eval.sh

# Multi-scale test-time augmentation
MULTI_SCALE=1 CHECKPOINT=checkpoints/.../best.pth bash scripts/eval.sh

# Save prediction visualizations
SAVE_VIS=1 CHECKPOINT=checkpoints/.../best.pth bash scripts/eval.sh
```

## Inference (Single Image)

```bash
# Default (uses text config from experiment YAML)
RGB=demo/rgb.jpg DEPTH=demo/depth.png CHECKPOINT=checkpoints/.../best.pth bash scripts/infer.sh

# Override text mode from CLI
python tools/infer.py --config configs/experiments/nyu_dtformer_s.yaml \
    --checkpoint checkpoints/.../best.pth \
    --rgb demo/rgb.jpg --depth demo/depth.png \
    --text-mode image_specific \
    --labels wall floor table chair
```

Output: a palette-colored PNG prediction map.

## Model Variants

| Variant | Params | FLOPs | Config |
|---------|--------|-------|--------|
| DTFormer-S | ~25M | ~40G | `configs/models/dtformer_s.yaml` |
| DTFormer-B | ~48M | ~80G | `configs/models/dtformer_b.yaml` |
| DTFormer-L | ~85M | ~161G | `configs/models/dtformer_l.yaml` |

All variants use TSA-E at encoder stages 1/2/3 and TSA-D at all decoder levels by default. Stage 0 is reserved for ablation (set `tsae_stages: [0,1,2,3]` in config to enable).

## Project Structure

```
dtformer/
├── configs/                    # YAML configuration files
│   ├── datasets/               #   Dataset definitions (NYU, SUNRGBD)
│   ├── models/                 #   Model architecture (S / B / L)
│   └── experiments/            #   Training recipes (dataset + model + hyperparams)
├── src/dtformer/               # Core library
│   ├── data/                   #   Datasets, transforms, collate, text store
│   ├── text/                   #   CLIP backend, templates, vocabularies, cache I/O
│   ├── models/
│   │   ├── backbones/          #     DTFormerEncoder (GSA + TSA-E)
│   │   ├── modules/            #     TSCA, geometry attention
│   │   ├── decoders/           #     HSG decoder (Hamburger + TSA-D)
│   │   └── segmentors/        #     Top-level DTFormer model
│   ├── engine/                 #   Train/eval/infer loops, optimizer, scheduler, metrics
│   └── utils/                  #   Logging, seed, I/O, env
├── tools/                      # CLI entry points
│   ├── train.py                #   Training (torchrun compatible)
│   ├── eval.py                 #   Evaluation
│   ├── infer.py                #   Single-image inference
│   ├── build_clip_cache.py     #   Pre-compute CLIP text embeddings
│   ├── build_image_labels.py   #   Extract per-image labels from GT masks
│   ├── merge_image_labels.py   #   Merge/intersect VLM label outputs
│   ├── generate_tags_internvl.py  # VLM label generation (InternVL3)
│   └── generate_tags_qwen.py     # VLM label generation (Qwen3-VL)
├── scripts/                    # Shell launch scripts
│   ├── train.sh
│   ├── eval.sh
│   └── infer.sh
├── research/                   # Non-core research utilities
│   ├── visualization/          #   Label overlay, performance plots
│   ├── analysis/               #   Label statistics, overlap comparison
│   └── paper_repro/            #   Paper reproduction helpers
├── requirements.txt
└── README.md
```

## Key Architectural Decisions

**TSA-E** (encoder, Eq. 6): Lightweight TSCA + gated residual, no FFN. Applied inside each RGB-D-T Block at enabled stages. Parameter sharing via configurable share factor per stage (see Table 9).

**TSA-D** (decoder, Eq. 7-8): Pre-LN + TSCA + MLP + gated residual. Applied per decoder level before upsampling.

**TSCA** (Eq. 4): L2-normalized Q/K with learnable temperature. Always on (cosine similarity + learnable temp hardcoded per Table 5 "Full" config).

**HSG decoder**: Per-level TSA-D, upsample, concatenate, Hamburger (NMF), classify.

**No mmseg/mmcv**: All replaced with native PyTorch equivalents (`F.interpolate`, `nn.SyncBatchNorm`, simple Conv+BN wrapper).

## Text Modes

**`fixed`**: Uses the full dataset vocabulary (e.g., all 40 NYU class names). Same text embeddings for every image.

**`image_specific`**: Per-image top-K labels (from VLM or GT), truncated to K tokens. Different text per image.

Configure in experiment YAML:
```yaml
text:
  mode: image_specific    # or "fixed"
  max_image_labels: 6
```
