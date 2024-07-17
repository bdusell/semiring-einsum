set -euo pipefail

. scripts/dockerdev.bash
. scripts/variables.bash

on_start() {
  dockerdev_run_in_dev_container "$IMAGE" bash -c '
    ln -s /home/dummy/.ssh "$HOME"/.ssh
  '
}

bash scripts/build-dev-image.bash
dockerdev_ensure_dev_container_started "$IMAGE" \
  --on-start on_start \
  -- \
  -v "$PWD":/app/ \
  --mount type=bind,source="$HOME"/.ssh/,destination=/home/dummy/.ssh/ \
  --gpus all
dockerdev_run_in_dev_container "$IMAGE" "$@"
