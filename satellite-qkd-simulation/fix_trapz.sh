#!/usr/bin/env bash

set -e

echo "🔍 Searching for np.trapz occurrences..."

# Directory root (default: current)
ROOT_DIR="${1:-.}"

# File temporaneo log
LOG_FILE="trapz_replacements.log"
> "$LOG_FILE"

# Trova file Python contenenti trapz
FILES=$(grep -rl "np\.trapz" "$ROOT_DIR" --include="*.py" || true)

if [ -z "$FILES" ]; then
    echo "✅ No occurrences of np.trapz found."
    exit 0
fi

echo "⚠️ Found occurrences in:"
echo "$FILES"

echo ""
echo "🔧 Applying replacements..."

for file in $FILES; do
    echo "Processing: $file"

    # Backup
    cp "$file" "$file.bak"

    # Replace
    sed -i 's/np\.trapz/np.trapezoid/g' "$file"

    echo "$file" >> "$LOG_FILE"
done

echo ""
echo "✅ Replacement completed."
echo "📄 Modified files logged in: $LOG_FILE"
echo "💾 Backup files: *.bak"

echo ""
echo "👉 If everything works, you can remove backups with:"
echo "rm \$(find . -name '*.bak')"
