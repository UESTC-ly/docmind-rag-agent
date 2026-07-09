#!/usr/bin/env bash
# DocMind cross-platform launcher for macOS/Linux.
# Windows users: run .\start.ps1 from PowerShell.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OS_NAME="$(uname -s 2>/dev/null || echo unknown)"

case "$OS_NAME" in
  Darwin)
    exec "$ROOT_DIR/scripts/start-macos-colima.sh" "$@"
    ;;
  Linux)
    exec "$ROOT_DIR/scripts/start-linux-docker.sh" "$@"
    ;;
  MINGW*|MSYS*|CYGWIN*)
    echo "Windows detected. Please run: powershell -ExecutionPolicy Bypass -File .\\start.ps1" >&2
    exit 1
    ;;
  *)
    echo "Unsupported OS: $OS_NAME" >&2
    echo "Use scripts/start-linux-docker.sh, scripts/start-macos-colima.sh, or scripts/start-windows.ps1 directly." >&2
    exit 1
    ;;
esac
