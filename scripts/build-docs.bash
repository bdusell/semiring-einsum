set -e
set -u

cd docs
poetry run make clean
poetry run make html
