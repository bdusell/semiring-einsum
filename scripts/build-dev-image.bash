set -e
set -u

. scripts/variables.bash
DOCKER_BUILDKIT=1 docker build "$@" -t "$IMAGE":latest -f Dockerfile-dev .
