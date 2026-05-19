#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

read -rp "Enter path to MIMIC-IV-NOTE note directory containing CSV files: " MIMIC_NOTE_DIR

IMPORT_SCRIPT="$SCRIPT_DIR/buildmimic_duckdb/duckdb/import_duckdb.sh"
OUTPUT_DB="${1:-mimic4_note.db}"

if ! compgen -G "$MIMIC_NOTE_DIR/*.csv*" > /dev/null; then
    echo "Invalid directory: expected .csv or .csv.gz files."
    exit 1
fi

(
    cd "$SCRIPT_DIR/buildmimic_duckdb/duckdb"
    bash ./import_duckdb.sh "$MIMIC_NOTE_DIR" "$SCRIPT_DIR/$OUTPUT_DB"
)