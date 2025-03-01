#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:rtx8000:1

#SBATCH --job-name="cite"
#SBATCH --output=citea100/%j.out
#SBATCH --error=citea100/%j.err


module purge

singularity exec --nv --overlay /scratch/ys6310/prompter/overlay-50G-10M.ext3:rw \
/scratch/ys6310/prompter/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
/bin/bash -c "

source /ext3/env.sh
conda activate prompter
python -u cora.py > load_test.txt 2>&1
"