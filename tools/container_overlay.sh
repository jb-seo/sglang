#!/usr/bin/env bash
# Print the bind mount that overlays this tree's serving code onto a container
# running the stock release, and check the two agree on a base.
#
#   ./tools/container_overlay.sh                # base = stock v0.5.18
#   ./tools/container_overlay.sh <base-commit>
#   ABS=1 ./tools/container_overlay.sh          # absolute host path
#
# One mount, not one per file: every serving change lives under
# python/sglang/srt, which holds source only. Mounting the parent
# python/sglang instead would shadow the build-generated _version.py.
#
# The base check is the point. A file written against a different base
# references symbols the container's other modules do not have, and that
# surfaces at scheduler start as an ImportError naming an unrelated module —
# nothing about it says "your mount is stale".
set -euo pipefail

BASE="${1:-71de97b264}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/sgl-workspace/sglang}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

git rev-parse --verify --quiet "$BASE^{commit}" >/dev/null || {
  echo "error: base commit '$BASE' not found in this repo" >&2
  exit 1
}
BASE_SHA="$(git rev-parse "$BASE")"

if [ "${ABS:-0}" = "1" ]; then
  HOST_PATH="$REPO_ROOT/python/sglang/srt"
else
  HOST_PATH="./python/sglang/srt"
fi

echo "base   $BASE_SHA  ($(git describe --tags --always "$BASE_SHA"))"
echo "head   $(git rev-parse --short HEAD)  $(git log -1 --format=%s)"

if ! git diff --quiet -- python/sglang/ || ! git diff --cached --quiet -- python/sglang/; then
  echo
  echo "WARNING: uncommitted changes under python/sglang/ — the container will"
  echo "         run something no commit describes."
fi

echo
echo "changed vs base:"
git diff --name-only "$BASE" HEAD -- python/sglang/srt/ | sed 's/^/  /'
outside=$(git diff --name-only "$BASE" HEAD -- python/sglang/ ':!python/sglang/srt/')
if [ -n "$outside" ]; then
  echo
  echo "not covered by the mount (outside srt/ — test helpers, not serving):"
  echo "$outside" | sed 's/^/  /'
fi

cat <<EOF

Clear stale bytecode once (a read-only mount cannot rewrite it):
  find python/sglang/srt -name __pycache__ -type d -exec rm -rf {} +

Check the container is on the same base:
  docker exec <container> git -C $CONTAINER_ROOT rev-parse HEAD
  -> must print $BASE_SHA

Mount:
  -v $HOST_PATH:$CONTAINER_ROOT/python/sglang/srt:ro \\
EOF
