set -e
set -u

poetry run bash -c 'PYTHONPATH=$PWD:$PYTHONPATH python "$@"' -- "$@"
