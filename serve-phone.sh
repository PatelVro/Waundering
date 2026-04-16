#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-4173}"

if command -v hostname >/dev/null 2>&1; then
  HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
else
  HOST_IP=""
fi

if [[ -z "${HOST_IP}" ]]; then
  HOST_IP="<your-computer-local-ip>"
fi

echo "Starting VoltOps site for local-network access..."
echo ""
echo "1) Make sure your phone and this computer are on the same Wi-Fi network."
echo "2) Open this URL on your phone:" 
echo "   http://${HOST_IP}:${PORT}"
echo ""
echo "Press Ctrl+C to stop."
echo ""

python3 -m http.server "${PORT}" --bind 0.0.0.0
