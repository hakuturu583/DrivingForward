#!/usr/bin/env bash
set -euo pipefail

TORCHSCRIPT_DIR="/workspace/DrivingForward/torchscript"
mkdir -p "${TORCHSCRIPT_DIR}"
chown -R "${UID:-1000}:${GID:-1000}" "${TORCHSCRIPT_DIR}"

if [ "$#" -eq 0 ]; then
  exec su -s /bin/bash appuser
fi

exec su -s /bin/bash appuser -c "$*"
