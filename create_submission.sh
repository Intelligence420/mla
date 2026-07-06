#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${SCRIPT_DIR}/submission.tar.xz"

tar -cJf "$OUTPUT" -C "$SCRIPT_DIR" tar/

echo "Created: $OUTPUT"
