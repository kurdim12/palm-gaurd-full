#!/usr/bin/env bash
# End-to-end demo: simulate an infested detection through the edge uploader and
# watch it (1) flip a tree to infested on the API and (2) raise an Arabic alert.
#
# Usage: scripts/demo.sh
# Requires: the API deps installed (make install or pip install -r packages/api/requirements.txt)
# and a baseline/TFLite model artifact (make baseline).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT=8077
API_URL="http://localhost:${PORT}"
TREE_ID="tree-001"  # seeded clean, so it visibly transitions clean -> infested
QUEUE="$(mktemp -d)/queue.jsonl"

cleanup() { kill "${API_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT

echo "==> Ensuring a model artifact exists (baseline)…"
( cd "$ROOT/packages/ml" && python3 -m palmguard_ml.cli data >/dev/null 2>&1 || true )
( cd "$ROOT/packages/ml" && python3 -m palmguard_ml.cli baseline >/dev/null 2>&1 || true )

echo "==> Starting API on :${PORT} (in-memory DB)…"
( cd "$ROOT/packages/api" && python3 -m uvicorn app.main:app --port "$PORT" --log-level warning ) &
API_PID=$!
for _ in $(seq 1 20); do curl -sf "$API_URL/health" >/dev/null 2>&1 && break; sleep 0.5; done

CLIP="$(find "$ROOT/data/raw/infested" -name '*.wav' 2>/dev/null | head -1 || true)"
[ -z "$CLIP" ] && { echo "No infested clip found; run 'make data' first." >&2; exit 1; }

echo "==> Sending 3 confident infested detections through the edge agent…"
for i in 1 2 3; do
  ( cd "$ROOT/packages/edge" && \
    EDGE_QUEUE_PATH="$QUEUE" EDGE_API_URL="$API_URL" EDGE_TREE_ID="$TREE_ID" \
    python3 run.py --sim "$CLIP" --once )
done

echo
echo "==> Tree status on the API:"
curl -s "$API_URL/api/v1/trees/${TREE_ID}" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('   status:',d['status'],'| streak:',d['infested_streak'])"

echo "==> Open alerts (Arabic-first):"
curl -s "$API_URL/api/v1/alerts?only_open=true" \
  | python3 -c "import sys,json;[print('   🚨',a['message_ar']) for a in json.load(sys.stdin)]"

echo
echo "==> Done. Start the dashboard ('make web') to see ${TREE_ID} as a red pin on the map."
cleanup; trap - EXIT
