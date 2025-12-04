set -euo pipefail

. scripts/dockerdev.bash
. scripts/variables.bash

if [[ -f /etc/NIXOS ]]; then
  gpu_options=(--device=nvidia.com/gpu=all)
else
  gpu_options=(--gpus all --privileged)
fi

bash scripts/build-min-version-image.bash "$@"
dockerdev_ensure_dev_container_started "$MIN_VERSION_IMAGE" -- "${gpu_options[@]}"
dockerdev_run_in_dev_container "$MIN_VERSION_IMAGE" bash scripts/run-tests.bash
