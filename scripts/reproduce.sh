#!/usr/bin/env bash
# Full study, in dependency order. Steps run sequentially so training times are not distorted by
# competing jobs. Roughly 3 hours on an 8 GB Apple M3. Each step's log goes to results/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p results/logs
run() { local name=$1; shift; echo "=== $name: $*"; "$PY" -u "$@" > "results/logs/$name.log" 2>&1; echo "    done"; }

run eda            scripts/run_eda.py
run tune           scripts/tune.py --seed 0
for s in 0 1 2 3 4; do run "bench_250k_seed$s" scripts/run_benchmark.py --subset-size 250000 --seed "$s"; done
run ablations      scripts/run_ablations.py --seeds 0 1 2 --feature-sets low all
run interpret      scripts/interpret.py --seed 0 --feature-sets all geo
for s in 0 1 2; do run "bench_50k_seed$s"  scripts/run_benchmark.py --subset-size 50000 --seed "$s"; done
for s in 0 1 2; do run "bench_1M_seed$s"   scripts/run_benchmark.py --subset-size 1000000 --seed "$s"; done
# Full 10.5M pool: the 47-column geo sets would need >8 GB RAM here, so only low and all.
run bench_full_seed0 scripts/run_benchmark.py --subset-size 10500000 --seed 0 \
    --models logreg xgboost mlp --feature-sets low all
run aggregate      scripts/aggregate.py
echo "All done."
