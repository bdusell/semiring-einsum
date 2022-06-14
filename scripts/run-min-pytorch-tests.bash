set -e
set -u
set -o pipefail

. scripts/dockerdev.bash
. scripts/variables.bash

bash scripts/build-min-pytorch-image.bash "$@"
dockerdev_ensure_dev_container_started "$MIN_PYTORCH_IMAGE"
dockerdev_run_in_dev_container "$MIN_PYTORCH_IMAGE" bash scripts/run-tests.bash
