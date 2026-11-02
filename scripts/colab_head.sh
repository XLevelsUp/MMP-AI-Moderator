#!/usr/bin/env bash
# Gate 1 head on a Google Colab T4 (Runtime -> Change runtime type -> T4 GPU): embed the dataset,
# train the head, then compare it with the rules on the hard-case TEST set.
# Unzip mmp_head.zip into /content (gives mmp-moderator-poc/, dataset/, hard_case_clean/), then:
#   !bash /content/mmp-moderator-poc/scripts/colab_head.sh
# About 15 minutes. Download /content/head_results.zip at the end (the head file, logs, per-file csv).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
DATASET="${1:-/content/dataset}"
HARD="${2:-/content/hard_case_clean}"
OUT=data/gate1_head

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
    echo "NO GPU: this notebook runs on a CPU. Runtime -> Change runtime type -> T4 GPU -> Save,"
    echo "then upload the zip again (changing the runtime wipes /content) and rerun this cell."
    exit 1
fi
for folder in "$DATASET" "$HARD"; do
    [ -d "$folder" ] || { echo "missing folder $folder (unzip mmp_head.zip into /content)"; exit 1; }
done
[ -f "$DATASET/labels.csv" ] || { echo "missing $DATASET/labels.csv"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv export --no-hashes --no-dev --format requirements-txt | grep -v -E "^(torch|torchvision|torchaudio)" > /tmp/req.txt
uv pip install --system -q -r /tmp/req.txt scikit-learn
python -c "import torchvision" 2>/dev/null || uv pip uninstall --system torchvision
python -c "import torchaudio" 2>/dev/null || uv pip uninstall --system torchaudio

python -m scripts.download_models --models siglip2_base owlv2 bioclip2

export ML_DEVICE=cuda ML_DTYPE=float16
mkdir -p "$OUT"
echo "== dry run on 3 photos (catches path / save errors in seconds)"
python -m scripts.head_embed "$DATASET" --limit 3 --out /tmp/head_dry.npz
echo "== 1/3 embeddings"
python -m scripts.head_embed "$DATASET" 2>&1 | tee "$OUT/embed.log"
echo "== 2/3 training"
python -m scripts.head_train 2>&1 | tee "$OUT/train.log"
echo "== 3/3 rules vs head on the hard-case set"
python -m scripts.head_eval "$HARD" 2>&1 | tee "$OUT/eval.log"

rm -f /content/head_results.zip
python -m zipfile -c /content/head_results.zip models/gate1_head.npz "$OUT"
echo "== done: download /content/head_results.zip"
