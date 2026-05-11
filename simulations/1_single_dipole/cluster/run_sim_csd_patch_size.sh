#!/bin/bash

# ----- #
# Job name
#SBATCH --job-name=patchsize_sim


# ----- #
# Jobs to run; each element corresponds to a subject.
# Only run n at a time with (%n) at the end of the command!
# Counting the python way!
#SBATCH --array=0 # --array=0%150


# ----- #
# Computational resources.
#SBATCH --cpus-per-task=8 
#SBATCH --mem=32G

# Instead of specifying a nodelist which will ask for all the nodes to be
# available for each job, exclude the nodes that are not contained in the
# nodelist. Each job will only occupy one node this way.
# SBATCH --nodelist=node[13-21]
# SBATCH --exclude=node[2-12]


# ----- #
# Task time limit (D-HH:MM:SS)
#SBATCH --time=1-00:00:00 # one task should last 4h


# ----- #
# Output and error filenames.
# Currently skipped and instead used directly when calling the python script.
# --output=fichier_de_sortie${SLURM_ARRAY_TASK_ID}.txt
# --error=sortie_erreur.err


# ----- #
# Python activation.
module add Programming_Languages/anaconda/3.11

# Activation of virtual python environment.
conda activate lameg

#SBATCH --licenses=sps

# ----- #
# Run script.
# Standard output and standard error are NOT redirected to the same file.
python -u /pbs/home/s/sgailhard/csd_simulations/sim_csd_patch_size.py > /sps/isc/sgailhard/csd_simulations/output/output_patch_size_$SLURM_ARRAY_TASK_ID.txt 2> /sps/isc/sgailhard/csd_simulations/output/error_patch_size_$SLURM_ARRAY_TASK_ID.txt ${SLURM_ARRAY_TASK_ID}