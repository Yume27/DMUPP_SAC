#!/bin/bash

# Allocation account name
#SBATCH --account=ucb516_asc2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=23:59:00
# Jobs to run, inclusive
#SBATCH --array=0-0
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --output=/scratch/alpine/%u/job_output/job_%a-%j.out
#SBATCH --mail-type=ALL
#SBATCH --dependency=singleton

# Copy this script as sweep.sh to modify it for your own use.

module purge

echo "Loading modules"
module load python/3.10.2
module load gcc

echo "Activating virtual environment"
source /projects/$USER/DMUPP/.venv_DMUPP/bin/activate
export TMPDIR=/scratch/alpine/$USER/temp_dir

echo "Running training script"
# Uses int(sys.argv[1]) in the script to get the array index
python3 /projects/$USER/DMUPP/code/Dreamer_train.py $SLURM_ARRAY_TASK_ID

echo "== End of Job =="
