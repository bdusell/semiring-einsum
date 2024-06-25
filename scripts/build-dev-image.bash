set -euo pipefail

. scripts/variables.bash
DOCKER_BUILDKIT=1 docker build "$@" -t "$IMAGE":latest -f Dockerfile-dev .
