#!/bin/bash

#SBATCH --chdir=./

#SBATCH --job-name=BC_EMBS
#SBATCH --mail-type=START,FAIL,END
#SBATCH --mail-user=kmgomezm@eafit.edu.co
#SBATCH --output=slurm-serial.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=slurm-serial.%j.err  # Stderr (%j expands to jobId)
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --partition=learning       # Partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-23:00:00



##### ENVIRONMENT CREATION #####
module load python/3.10_miniconda-23.5.2

##### JOB COMMANDS #### 
export HF_TOKEN="hf_xxxx" # <<< tu token aquí

pip install --upgrade pip
pip install sentence-transformers pandas numpy huggingface-hub openpyxl pyarrow


# Ejecutar el script de Python # arreglar
python /home/mst-kmgomezm/hhdd/4_busqueda_ciencia.py