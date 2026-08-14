#!/bin/bash
set -e

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <path>"
  exit 1
fi

SEARCHPATH="$1"
SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
mapfile -d '' FOLDERS < <(find "$SEARCHPATH" -mindepth 1 -maxdepth 1 -type d -print0)

echo "Found ${#FOLDERS[@]} folders!"

for FOLDER in "${FOLDERS[@]}"
do
  echo "Exporting $FOLDER to ONNX"
  python $SCRIPT_DIR/export_onnx.py $FOLDER
done



