set -euo pipefail

cd docs
poetry run make clean
poetry run make html
