. scripts/dockerdev.bash &&
. scripts/build-dev-image.bash &&
dockerdev_ensure_dev_container_started "$IMAGE" -v "$PWD":/app/ --gpus all &&
dockerdev_run_in_dev_container "$IMAGE" bash
