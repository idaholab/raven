#/bin/bash

# Check if the right number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 SIM_PATH POSTPROCESSING_PATH LOCAL_NPROCS"
    exit 1
fi

SIM_PATH=$1
POSTPROCESSING_PATH=$2
LOCAL_NPROCS=$3
# Run the Python script with the provided simulation path
python3 get_good_sim_runs.py --dirpath "$SIM_PATH" --save_dir "$POSTPROCESSING_PATH" --nprocs "$LOCAL_NPROCS"

# Run the Slurm script with the provided simulation path and postprocessing path
sbatch bitterroot_slurm.sh -d "$SIM_PATH" -b 50 --save_dir "$POSTPROCESSING_PATH"

# Run the Python postprocessing script
python3 MFIX_output_info.py "$SIM_PATH" "$POSTPROCESSING_PATH"