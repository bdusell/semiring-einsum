set -e
set -u
set -o pipefail

. scripts/dockerdev.bash
. scripts/variables.bash

bash scripts/build-min-version-image.bash "$@"
dockerdev_ensure_dev_container_started "$MIN_VERSION_IMAGE" -- --gpus all
dockerdev_run_in_dev_container "$MIN_VERSION_IMAGE" bash scripts/run-tests.bash
