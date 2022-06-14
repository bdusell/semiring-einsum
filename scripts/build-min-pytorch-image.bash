set -e
set -u

. scripts/variables.bash

DOCKER_BUILDKIT=1 docker build "$@" \
  --build-arg MIN_PYTHON_VERSION="$(< MIN_PYTHON_VERSION)" \
  -t "$MIN_PYTORCH_IMAGE":latest \
  -f Dockerfile-min-pytorch \
  .
