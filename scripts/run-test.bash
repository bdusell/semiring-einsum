set -e
set -u

PYTHONPATH=$PWD:${PYTHONPATH-} python "$@"
