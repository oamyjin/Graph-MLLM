#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=320GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="gnnbased"
#SBATCH --account=pr_142_tandon_advanced

module purge

singularity exec --nv --overlay /scratch/jl11523/mmSing/overlay-50G-10M.ext3:ro \
    /scratch/jl11523/mmSing/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "

source /ext3/env.sh
conda activate mm_bench2

# Declare model types
MODEL_LIST=('MLP' 'GCN' 'SAGE')

# Declare feature types
FEAT_LIST=('clip' 'clipimage' 'clipnonstruc' 'clipnonstruc')

# Declare datasets
DATASET_LIST=('Movies' 'Toys' 'Grocery' 'Arts' 'CD')

# Merge the results
RESULT_FILE=${SLURM_JOB_ID}_results.log
touch \$RESULT_FILE

# Loop over each model type
for DATASET in \"\${DATASET_LIST[@]}\"
do
    # Loop over each feature type
    for MODEL in \"\${MODEL_LIST[@]}\"
    do
        # Loop over each dataset
        for FEAT in \"\${FEAT_LIST[@]}\"
        do
            python main.py --config-name defaults \
                    model_name=\$MODEL \
                    dataset=\$DATASET \
                    result_file=\$RESULT_FILE \
                    feat=\$FEAT > ${SLURM_JOB_ID}_\$DATASET-\$MODEL-\$FEAT.log 2>&1
        done
    done
done
"
