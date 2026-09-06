#!/usr/bin/env bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TAG="surgvu26-cat2-algo"
source "$SCRIPT_DIR/do_build.sh"
mkdir -p "$SCRIPT_DIR/test/output/interf0" && rm -rf "$SCRIPT_DIR/test/output/interf0"/*
docker volume create "$TAG-tmp" > /dev/null
docker run --rm --platform=linux/amd64 --network none \
  --volume "$SCRIPT_DIR/test/input/interf0":/input:ro \
  --volume "$SCRIPT_DIR/test/output/interf0":/output \
  --volume "$TAG-tmp":/tmp \
  --volume "$SCRIPT_DIR/model":/opt/ml/model:ro \
  "$TAG"
docker volume rm "$TAG-tmp" > /dev/null
echo "=== OUTPUT ===" && ls -la "$SCRIPT_DIR/test/output/interf0/"
