#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-4173}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is required for public sharing."
  echo "Install: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
  exit 1
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${TUNNEL_PID:-}" ]] && kill -0 "${TUNNEL_PID}" >/dev/null 2>&1; then
    kill "${TUNNEL_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

python3 -m http.server "${PORT}" --bind 127.0.0.1 >/tmp/voltops-local.log 2>&1 &
SERVER_PID=$!

sleep 1

LOG_FILE="/tmp/voltops-cloudflared.log"
cloudflared tunnel --url "http://127.0.0.1:${PORT}" --no-autoupdate >"${LOG_FILE}" 2>&1 &
TUNNEL_PID=$!

echo "Starting secure public tunnel..."
echo ""
echo "When the URL appears below, open it on your phone from any network."
echo "Press Ctrl+C to stop sharing."
echo ""

for _ in $(seq 1 60); do
  if grep -Eo 'https://[-a-zA-Z0-9]+\.trycloudflare\.com' "${LOG_FILE}" >/dev/null 2>&1; then
    grep -Eo 'https://[-a-zA-Z0-9]+\.trycloudflare\.com' "${LOG_FILE}" | head -n 1
    break
  fi
  sleep 1
done

echo ""
tail -f "${LOG_FILE}"
