#!/usr/bin/env bash
# One-shot setup: installs dependencies and downloads all required files.
# Usage:  bash setup.sh [--sam-model vit_b|vit_l|vit_h]
set -euo pipefail

SAM_MODEL="vit_b"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sam-model) SAM_MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ "$SAM_MODEL" == "vit_l" ]]; then
    SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
elif [[ "$SAM_MODEL" == "vit_h" ]]; then
    SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
else
    SAM_MODEL="vit_b"
    SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
fi
SAM_FILE="checkpoints/$(basename "$SAM_URL")"

echo "============================================================"
echo " HLCV Project Setup"
echo " SAM model: $SAM_MODEL"
echo "============================================================"

# ── 1. Python dependencies ────────────────────────────────────────
echo ""
echo "[1/4] Installing Python dependencies..."
pip install -q -r requirements.txt
pip install -q git+https://github.com/facebookresearch/segment-anything.git
pip install -q -e .
echo "      Done."

# ── 2. Create directories ────────────────────────────────────────
echo ""
echo "[2/4] Creating directories..."
mkdir -p data/coco/val2017 data/coco/annotations \
         data/subsets checkpoints results figures notebooks report
echo "      Done."

# ── 3. Parallel downloads ─────────────────────────────────────────
echo ""
echo "[3/4] Downloading COCO val2017 + SAM checkpoint in parallel..."
echo "      This may take several minutes depending on your connection."

COCO_IMG_ZIP="data/coco/val2017.zip"
COCO_ANN_ZIP="data/coco/annotations_trainval2017.zip"

download() {
    local url="$1"
    local dest="$2"
    local label="$3"
    if [[ -f "$dest" ]]; then
        echo "      [skip] $label already exists."
    else
        echo "      Downloading $label ..."
        wget -q --show-progress -O "$dest" "$url"
        echo "      [done] $label"
    fi
}

# Launch three downloads in background
download "http://images.cocodataset.org/zips/val2017.zip" \
         "$COCO_IMG_ZIP" "COCO val2017 images" &
PID_IMG=$!

download "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
         "$COCO_ANN_ZIP" "COCO annotations" &
PID_ANN=$!

download "$SAM_URL" "$SAM_FILE" "SAM $SAM_MODEL checkpoint" &
PID_SAM=$!

# Wait for all three
wait $PID_IMG && echo "      [done] COCO images download finished." || echo "      [WARN] COCO images download failed."
wait $PID_ANN && echo "      [done] COCO annotations download finished." || echo "      [WARN] COCO annotations download failed."
wait $PID_SAM && echo "      [done] SAM checkpoint download finished." || echo "      [WARN] SAM checkpoint download failed."

# ── 4. Extract archives ───────────────────────────────────────────
echo ""
echo "[4/4] Extracting archives..."

if [[ -f "$COCO_IMG_ZIP" ]]; then
    echo "      Extracting COCO images..."
    unzip -q -o "$COCO_IMG_ZIP" -d data/coco/
    echo "      [done] Images extracted → data/coco/val2017/"
fi

if [[ -f "$COCO_ANN_ZIP" ]]; then
    echo "      Extracting COCO annotations..."
    unzip -q -o "$COCO_ANN_ZIP" -d data/coco/
    echo "      [done] Annotations extracted → data/coco/annotations/"
fi

echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Build evaluation subset (200 images):"
echo "      python src/dataset/coco_loader.py \\"
echo "             --n-images 200 \\"
echo "             --output data/subsets/subset_200.json"
echo ""
echo "   2. Run SAM evaluation:"
echo "      python src/experiments/run_sam.py \\"
echo "             --subset data/subsets/subset_200.json \\"
echo "             --checkpoint $SAM_FILE"
echo ""
echo "   3. Run Mask R-CNN evaluation:"
echo "      python src/experiments/run_maskrcnn.py \\"
echo "             --subset data/subsets/subset_200.json"
echo ""
echo "   4. Run Mask2Former evaluation:"
echo "      python src/experiments/run_mask2former.py \\"
echo "             --subset data/subsets/subset_200.json"
echo ""
echo "   5. Generate all plots:"
echo "      python src/visualization/plot.py \\"
echo "             --sam results/sam_results.json \\"
echo "             --maskrcnn results/maskrcnn_results.json \\"
echo "             --mask2former results/mask2former_results.json \\"
echo "             --output figures/"
echo "============================================================"
