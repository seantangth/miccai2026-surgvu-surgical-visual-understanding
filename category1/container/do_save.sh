#!/usr/bin/env bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TAG="surgvu26-cat1-algo"
source "$SCRIPT_DIR/do_build.sh"
OUT="${SURGVU_OUT:-./submissions}/${TAG}_$(date +%Y%m%d-%H%M%S).tar.gz"
echo "Saving image to $OUT (takes a while)..."
docker save "$TAG" | gzip -c > "$OUT"
echo "Saved: $OUT ($(du -h "$OUT" | cut -f1))"
