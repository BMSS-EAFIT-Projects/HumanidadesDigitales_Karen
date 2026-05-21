# =============================================================================
# BÚSQUEDA DE CIENCIA EN CHUNKS - Embeddings + Matriz de similitud
# =============================================================================

import os
import glob
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# =============================================================================
# 1. RUTAS Y PARÁMETROS — AJUSTAR SEGÚN EL ENTORNO
# =============================================================================

PATH_CHUNKS_PARQUET   = r"/home/mst-kmgomezm/hhdd/data/processed/chunks.parquet"              # <<< cambiar
DIR_TESAURO           = r"/home/mst-kmgomezm/hhdd/data/external/terminos/Tesauro_Unesco_Ciencia" # <<< cambiar
PATH_CHUNK_EMBEDDINGS = r"/home/mst-kmgomezm/hhdd/data/processed/chunk_embeddings.npy"             # <<< cambiar
DIR_RESULTS           = r"/home/mst-kmgomezm/hhdd/data/results"                                  # <<< cambiar

MODEL_ID   = "google/embeddinggemma-300M"
BATCH_SIZE = 8
RELOAD     = False   # True = recalcular embeddings aunque ya existan

# Credenciales Hugging Face

# Opción B: variable de entorno antes de correr el script
#   Linux/Mac:  export HF_TOKEN="hf_xxxx"
#   Windows:    set HF_TOKEN=hf_xxxx
# Opción C: pegar el token directamente (no subir a git)
# HF_TOKEN = "hf_xxxx"  # <<< pegar token aquí (opcional, no subir a git)
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# =============================================================================
# 2. AUTENTICACIÓN HUGGING FACE
# =============================================================================

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
    print("Login en Hugging Face exitoso.")
else:
    print("HF_TOKEN no encontrado — solo modelos públicos disponibles.")

# =============================================================================
# 3. CARGA DE DATOS Y MODELO
# =============================================================================

chunks_df = pd.read_parquet(PATH_CHUNKS_PARQUET)
print(f"Chunks cargados: {len(chunks_df):,} | Columnas: {chunks_df.columns.tolist()}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = SentenceTransformer(MODEL_ID, token=HF_TOKEN).to(device=device)
print(f"Modelo cargado | Max seq length: {model.max_seq_length}")

# =============================================================================
# 4. FUNCIONES
# =============================================================================

def cargar_subcategorias(path_dir: str, model: SentenceTransformer) -> dict:
    """
    Lee los .txt del directorio de subcategorías, concatena los términos
    de cada archivo en una sola frase y devuelve un embedding por subcategoría.

    Returns:
        dict {nombre_subcat: tensor_embedding}
    """
    subcat_embeddings = {}

    for file_path in glob.glob(os.path.join(path_dir, "*.txt")):
        subcat = os.path.splitext(os.path.basename(file_path))[0]
        with open(file_path, "r", encoding="utf-8") as f:
            terms = [line.strip() for line in f if line.strip()]
        phrase = " ".join(terms)
        subcat_embeddings[subcat] = model.encode(
            phrase, convert_to_tensor=True, show_progress_bar=False
        )

    print(f"Subcategorías cargadas ({len(subcat_embeddings)}): {list(subcat_embeddings.keys())}")
    return subcat_embeddings


def obtener_embeddings_chunks(
    chunks_df: pd.DataFrame,
    model: SentenceTransformer,
    batch_size: int = 64,
    save_path: str = PATH_CHUNK_EMBEDDINGS,
    reload: bool = False,
) -> np.ndarray:
    """
    Calcula embeddings de los chunks o los carga desde disco si ya existen.

    Returns:
        np.ndarray de shape (n_chunks, dim_embedding).
    """
    if not reload and os.path.exists(save_path):
        print(f"Cargando embeddings desde '{save_path}'...")
        embeddings = np.load(save_path)
        print(f"  Shape: {embeddings.shape}")
        return embeddings

    print("Calculando embeddings desde cero...")
    textos = chunks_df["texto_chunk"].tolist()
    chunk_embeddings = model.encode(
        textos,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
        prompt_name="STS",
    )
    embeddings_np = chunk_embeddings.cpu().numpy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, embeddings_np)
    print(f"  Guardados en '{save_path}' | Shape: {embeddings_np.shape}")
    return embeddings_np


def calcular_similitudes_chunks(
    chunks_df: pd.DataFrame,
    chunk_embeddings: np.ndarray,
    subcat_embeddings: dict,
) -> pd.DataFrame:
    """
    Calcula la similitud coseno entre cada chunk y cada subcategoría.

    Returns:
        DataFrame con columnas: chunk_id, id_doc, texto_chunk, <subcats...>
    """
    chunk_tensor  = torch.tensor(chunk_embeddings)
    subcats       = list(subcat_embeddings.keys())
    subcat_matrix = torch.stack(list(subcat_embeddings.values()))

    sim_matrix = util.cos_sim(chunk_tensor, subcat_matrix).cpu().numpy()

    sim_df = pd.DataFrame(sim_matrix, columns=subcats)
    sim_df.insert(0, "chunk_id",    chunks_df["chunk_id"].values)
    sim_df.insert(1, "id_doc",      chunks_df["id_doc"].values)
    sim_df.insert(2, "texto_chunk", chunks_df["texto_chunk"].values)

    return sim_df

# =============================================================================
# 5. PIPELINE
# =============================================================================

# Embeddings del tesauro (uno por subcategoría)
subcat_embeddings = cargar_subcategorias(DIR_TESAURO, model)

# Embeddings de los chunks
chunk_embeddings = obtener_embeddings_chunks(
    chunks_df,
    model,
    batch_size=BATCH_SIZE,
    save_path=PATH_CHUNK_EMBEDDINGS,
    reload=RELOAD,
)

# Matriz de similitud coseno
sim_df = calcular_similitudes_chunks(chunks_df, chunk_embeddings, subcat_embeddings)
print(f"Matriz de similitud: {sim_df.shape}")

# Guardar resultados
os.makedirs(DIR_RESULTS, exist_ok=True)
path_out = os.path.join(DIR_RESULTS, "similitudes_chunks.xlsx")
sim_df.to_excel(path_out, index=False, engine="openpyxl")
print(f"Resultados guardados en '{path_out}'")