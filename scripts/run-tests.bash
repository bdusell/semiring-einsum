set -e
set -u
set -o pipefail

find tests -name 'test_*.py' | sort | while read -r line; do
  bash scripts/run-test.bash "$line"
done
