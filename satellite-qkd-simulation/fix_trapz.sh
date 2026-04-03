#!/usr/bin/env bash

set -e

echo "=== Fix numpy trapz -> trapezoid ==="

# Directory progetto (modifica se necessario)
ROOT_DIR="."

# Backup directory
BACKUP_DIR="./backup_trapz_$(date +%s)"
mkdir -p "$BACKUP_DIR"

echo "Backup in: $BACKUP_DIR"
echo ""

# Trova file con trapz
FILES=$(grep -rl "np.trapz" "$ROOT_DIR" --include="*.py")

if [ -z "$FILES" ]; then
    echo "Nessun np.trapz trovato. Fine."
    exit 0
fi

echo "File trovati:"
echo "$FILES"
echo ""

for file in $FILES; do
    echo "Processing: $file"

    # backup
    cp "$file" "$BACKUP_DIR/$(basename "$file")"

    # sostituzione sicura
    sed -i 's/np\.trapz/np.trapezoid/g' "$file"

    echo "✔ aggiornato"
    echo ""
done

echo "=== COMPLETATO ==="
echo "Backup salvato in: $BACKUP_DIR"
