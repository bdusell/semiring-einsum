IMAGE=semiring-einsum-dev
DOCKER_BUILDKIT=1 docker build "$@" -t "$IMAGE":latest -f Dockerfile-dev .
