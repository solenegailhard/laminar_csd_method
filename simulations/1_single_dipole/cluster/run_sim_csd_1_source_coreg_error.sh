#!/bin/bash

# ----- #
# Job name
#SBATCH --job-name=err_sim


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
#SBATCH --exclude=ccwslurm0368,ccwslurm0369,ccwslurm0370,ccwslurm0371,ccwslurm0372


# ----- #
# Task time limit (D-HH:MM:SS)
#SBATCH --time=1-00:00:00 # one task should last 4h


# ----- #
# Output and error filenames.
# Currently skipped and instead used directly when calling the python script.
# --output=fichier_de_sortie${SLURM_ARRAY_TASK_ID}.txt
# --error=sortie_erreur.err

# clean job cache
export FONTCONFIG_PATH=/etc/fonts
export FONTCONFIG_FILE=/etc/fonts/fonts.conf
export XDG_CACHE_HOME=/tmp/$USER/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/fontcache
mkdir -p "$XDG_CACHE_HOME"
fc-cache -r >/dev/null 2>&1

export MCR_CACHE_ROOT=/tmp/$USER/mcr_cache_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$MCR_CACHE_ROOT"

cleanup() {
    rm -rf "$MCR_CACHE_ROOT" "$XDG_CACHE_HOME"
}
trap cleanup EXIT

# ----- #
# Python activation.
module add Programming_Languages/anaconda/3.11

# Activation of virtual python environment.
conda activate lameg

#SBATCH --licenses=sps

# ----- #
# Run script.
# Standard output and standard error are NOT redirected to the same file.
python -u /pbs/home/s/sgailhard/csd_simulations/sim_csd_1_source_coreg_error.py > /sps/isc/sgailhard/csd_simulations/output/output_1_source_coreg_error_$SLURM_ARRAY_TASK_ID.txt 2> /sps/isc/sgailhard/csd_simulations/output/error_1_source_coreg_error_$SLURM_ARRAY_TASK_ID.txt ${SLURM_ARRAY_TASK_ID}