#!/usr/bin/env bash
# 只做 tarball，不上傳、不提交（上傳會改 GC Active model，需用戶同意）。
# 用法：bash make_model_tarball.sh <gdrive run 名，如 v7_rA_rtdetr-l> <tag> [best|last]
set -euo pipefail
RUN="$1"; TAG="$2"; WHICH="${3:-last}"
ROOT="${SURGVU_ROOT:-$(pwd)}"
mkdir -p "$ROOT/4_models/v7/$TAG"
rclone copy "gdrive:MICCAI_2026_SurgVU/runs/$RUN/$WHICH.pt" "$ROOT/4_models/v7/$TAG/" 2>/dev/null
rclone copy "gdrive:MICCAI_2026_SurgVU/runs/$RUN/results.csv" "$ROOT/4_models/v7/$TAG/" 2>/dev/null || true
TMP=$(mktemp -d); cp "$ROOT/4_models/v7/$TAG/$WHICH.pt" "$TMP/best.pt"
OUT="$ROOT/5_outputs/submissions/cat1_model_${TAG}_$(date +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$OUT" -C "$TMP" .; rm -rf "$TMP"
ls -la "$OUT"; shasum -a 256 "$ROOT/4_models/v7/$TAG/$WHICH.pt" | cut -c1-16
