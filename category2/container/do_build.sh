#!/usr/bin/env bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
docker build --platform=linux/amd64 --tag "surgvu26-cat2-algo" "$SCRIPT_DIR"
