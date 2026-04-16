#!/bin/bash

#SBATCH --chdir=./
#SBATCH --job-name=EE_COL
#SBATCH --mail-type=START,FAIL,END
#SBATCH --mail-user=kmgomezm@eafit.edu.co
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --partition=accel-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-23:00:00

# ── Rutas del proyecto ────────────────────────────────────────────────────────
PROJECT_DIR="/home/kmgomezm/mySpace"
SCRIPT="$PROJECT_DIR/scripts/busqueda_ciencia/busqueda_ciencia.py"

CHUNKS_PATH="$PROJECT_DIR/data/raw/chunks.parquet"
TESAURO_DIR="$PROJECT_DIR/data/external/terminos/Tesauro_Unesco_Ciencia"
EMBEDDINGS_PATH="$PROJECT_DIR/data/processed/chunk_embeddings.npy"
RESULTS_DIR="$PROJECT_DIR/data/results"
FIGURES_DIR="$PROJECT_DIR/reports/figures"

MODEL_ID="google/embeddinggemma-300M"
BATCH_SIZE=64
UMBRAL=0.4

# ── Entorno ───────────────────────────────────────────────────────────────────
module load python/3.10_miniconda-23.5.2

conda activate busqueda_ciencia   # <-- cambia por el nombre de tu environment

# ── Logging ───────────────────────────────────────────────────────────────────
mkdir -p logs
echo "=============================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Nodo     : $SLURMD_NODENAME"
echo "Inicio   : $(date)"
echo "GPU info :"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================="


# ── Ejecución ─────────────────────────────────────────────────────────────────
python "$SCRIPT" \
    --chunks_path      "$CHUNKS_PATH"      \
    --tesauro_dir      "$TESAURO_DIR"      \
    --embeddings_path  "$EMBEDDINGS_PATH"  \
    --results_dir      "$RESULTS_DIR"      \
    --figures_dir      "$FIGURES_DIR"      \
    --model_id         "$MODEL_ID"         \
    --batch_size       $BATCH_SIZE         \
    --umbral           $UMBRAL
    # Agrega --reload si quieres forzar el recálculo de embeddings

echo "=============================="
echo "Fin: $(date)"
echo "=============================="