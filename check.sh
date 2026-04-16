#!/usr/bin/env bash
set -euo pipefail

for file in index.html styles.css script.js serve-phone.sh share-public.sh README.md; do
  [[ -f "$file" ]] || { echo "Missing required file: $file"; exit 1; }
done

grep -q "VoltOps ERP" index.html
grep -q "name=\"description\"" index.html
grep -q "new Lenis" script.js
grep -q "View it on your phone" README.md

echo "Static checks passed."
