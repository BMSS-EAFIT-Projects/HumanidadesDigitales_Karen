#!/bin/bash

#SBATCH --chdir=./

#SBATCH --job-name=FREC_POS
#SBATCH --mail-type=START,FAIL,END
#SBATCH --mail-user=kmgomezm@eafit.edu.co
#SBATCH --output=slurm-serial.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=slurm-serial.%j.err  # Stderr (%j expands to jobId)
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --partition=accel-2       # Partition
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-4:00:00



##### ENVIRONMENT CREATION #####
module load python/3.10_miniconda-23.5.2

##### JOB COMMANDS #####
 
pip install --upgrade pip
pip install stanza pandas tqdm openpyxl
 
python /home/mst-kmgomezm/hhdd/10_extraer_frecuencias_pos.py
 