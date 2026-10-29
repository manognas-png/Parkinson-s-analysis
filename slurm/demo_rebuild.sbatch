#!/bin/bash
# Rebuild the demo after the weight-estimator change, in order (stops at the first failure):
#   1. template + insole weight calibration and accuracy   (demo/build_template.py)
#   2. group medians, curves, pairs, clips                 (demo/build_demo_data.py)
#   3. JavaScript engine vs Python                          (demo/validate_engine.py)
#   4. single-file page                                     (demo/build_demo.py)
#   5. page smoke test                                      (demo/test_demo.js)
# Submit from the project root:
#   sbatch --account=bioinfmath540f26_class slurm/demo_rebuild.sbatch

#SBATCH --job-name=demo_rebuild
#SBATCH --account=bioinfmath540f26_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
module load node.js/22.14.0
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "===== 1. template and weight calibration ====="
python demo/build_template.py
echo "===== 2. demo data ====="
python demo/build_demo_data.py
echo "===== 3. validate engine against Python ====="
python demo/validate_engine.py
echo "===== 4. build page ====="
python demo/build_demo.py
echo "===== 5. smoke test ====="
node demo/test_demo.js
