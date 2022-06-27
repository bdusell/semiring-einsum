set -e
set -u
set -o pipefail

. scripts/variables.bash

get_min_version() {
  local name=$1
  grep "$name = " pyproject.toml | sed 's/.*"^\(.*\)"/\1/'
}

min_python_version=$(get_min_version python)
min_pytorch_version=$(get_min_version torch)
DOCKER_BUILDKIT=1 docker build "$@" \
  --build-arg MIN_PYTHON_VERSION="$min_python_version" \
  --build-arg MIN_PYTORCH_VERSION="$min_pytorch_version" \
  -t "$MIN_VERSION_IMAGE":latest \
  -f Dockerfile-min-version \
  .
