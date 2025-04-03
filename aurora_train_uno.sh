#!/bin/bash -x

# This script is executed by mpiexec for each rank.
# Determines Global Rank, Local Rank, calculates Target GPU/Tile,
# sets ZE_AFFINITY_MASK, and executes h.py passing rank, gpu, and tile info.

# --- Determine Global Rank ID ---
GLOBAL_RANK_ID="-1"
# Using PALS_RANKID as established previously
if [[ -n "$PALS_RANKID" ]]; then
  GLOBAL_RANK_ID="$PALS_RANKID"
else
  echo "Error (Global Rank): aurora_train_uno.sh could not determine Global MPI rank ID from PALS_RANKID." >&2
  exit 1
fi

# --- Determine Local Rank ID ---
LOCAL_RANK_ID="-1"
if [[ -n "$PALS_LOCAL_RANKID" ]]; then
  LOCAL_RANK_ID="$PALS_LOCAL_RANKID"
else
  echo "Error (Local Rank): aurora_train_uno.sh could not determine Local MPI rank ID from PALS_LOCAL_RANKID." >&2
  exit 1
fi

# --- Calculate Target GPU Tile and Set Affinity Mask ---
# Assuming 6 GPUs per node, 2 tiles per GPU, mapping local rank 0-11
TARGET_GPU=$(( LOCAL_RANK_ID / 2 ))
TARGET_TILE=$(( LOCAL_RANK_ID % 2 ))
#AFFINITY_MASK="${TARGET_GPU}.${TARGET_TILE}"
#export ZE_AFFINITY_MASK="${AFFINITY_MASK}" # Set for runtime/driver level affinity

export ZE_AFFINITY_MASK=0,1,2,3,4,5
echo "Rank ${GLOBAL_RANK_ID} (Local ${LOCAL_RANK_ID}): Calculated Target GPU=${TARGET_GPU}, Tile=${TARGET_TILE}. Setting ZE_AFFINITY_MASK=${AFFINITY_MASK}"

# --- Execute the Python Script ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# --- Environment Setup ---
echo "Setting up environment..."
module load frameworks || { echo "Error: Failed to load 'frameworks' module."; exit 1; }
source /home/rjain/venv/bin/activate || { echo "Error: Failed to activate virtual environment."; exit 1; }
# --- End Environment Setup ---

echo "Python version: $(python --version)"
echo "Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Current working directory: $(pwd)"
echo "Python executable: $(which python)"

PYTHON_EXE="$CONDA_PREFIX/bin/python"
export PYTHONPATH=/home/rjain/IMPROVE

OUTPUT_DIR="result${TARGET_GPU}_${TARGET_TILE}"
OUTPUT_DIR=$(echo $OUTPUT_DIR | tr -d '[:space:]')  # Remove any accidental whitespace

echo "Rank ${GLOBAL_RANK_ID}: Launching ${PYTHON_EXE} ./uno_train_improve.py --input_dir exp_result --output_dir ${OUTPUT_DIR}"

# Execute python, passing global rank, target gpu, and target tile as arguments
# Any extra arguments ($@) received by this script are passed at the end
# Corrected execution line:
"$PYTHON_EXE" ./uno_train_improve.py \
    --input_dir exp_result \
    --output_dir "$OUTPUT_DIR" \
    "$@"

EXIT_CODE=$?
echo "Rank ${GLOBAL_RANK_ID}: Python script finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}

