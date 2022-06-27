set -e
set -u

bash scripts/run-test.bash tests/plot_cost.py \
  --type time \
  tests/plot-data.json \
  docs/time-complexity.png \
  --block-sizes 1 2 3 4 5 10 20 50 \
  --no-pytorch
bash scripts/run-test.bash tests/plot_cost.py \
  --type time \
  tests/plot-data.json \
  docs/time-complexity-2.png
bash scripts/run-test.bash tests/plot_cost.py \
  --type space \
  tests/plot-data.json \
  docs/space-complexity.png \
  --block-sizes 50
bash scripts/run-test.bash tests/plot_cost.py \
  --type space \
  tests/plot-data.json \
  docs/space-complexity-2.png \
  --block-sizes 1 2 3 4 5 10 20 50 \
  --no-unbounded \
  --no-pytorch
