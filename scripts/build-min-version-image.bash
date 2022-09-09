set -e
set -u
set -o pipefail

. scripts/variables.bash

get_min_version() {
  local name=$1
  grep -m 1 "$name = " pyproject.toml | sed 's/.*"^\(.*\)"/\1/'
}

min_python_version=$(get_min_version python)
min_pynvml_version=$(get_min_version pynvml)
min_pytorch_version=$(get_min_version torch)
min_typing_extensions_version=$(get_min_version typing-extensions)

DOCKER_BUILDKIT=1 docker build "$@" \
  --build-arg MIN_PYTHON_VERSION="$min_python_version" \
  --build-arg MIN_PYNVML_VERSION="$min_pynvml_version" \
  --build-arg MIN_PYTORCH_VERSION="$min_pytorch_version" \
  --build-arg MIN_TYPING_EXTENSIONS_VERSION="$min_typing_extensions_version" \
  -t "$MIN_VERSION_IMAGE":latest \
  -f Dockerfile-min-version \
  .
