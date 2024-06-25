set -euo pipefail

PYTHONPATH=$PWD:${PYTHONPATH-} python "$@"
