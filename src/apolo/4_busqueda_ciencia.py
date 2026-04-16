"""
busqueda_ciencia.py
-------------------
Clasificación semántica de chunks de texto usando sentence-transformers
y un tesauro de subcategorías científicas (UNESCO).

Uso:
    python busqueda_ciencia.py \
        --chunks_path  /ruta/chunks.parquet \
        --tesauro_dir  /ruta/Tesauro_Unesco_Ciencia \
        --embeddings_path /ruta/chunk_embeddings.npy \
        --results_dir  /ruta/results \
        --model_id     google/embeddinggemma-300M \
        --batch_size   64 \
        --umbral       0.4 \
        --reload
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Búsqueda semántica de ciencia en chunks")

    parser.add_argument("--chunks_path",      required=True,  help="Ruta al archivo chunks.parquet")
    parser.add_argument("--tesauro_dir",       required=True,  help="Directorio con .txt de subcategorías")
    parser.add_argument("--embeddings_path",   required=True,  help="Ruta para guardar/cargar chunk_embeddings.npy")
    parser.add_argument("--results_dir",       required=True,  help="Directorio de salida para resultados")
    parser.add_argument("--figures_dir",       default=None,   help="Directorio de salida para figuras (default: results_dir/figures)")
    parser.add_argument("--model_id",          default="google/embeddinggemma-300M", help="ID del modelo en HuggingFace")
    parser.add_argument("--batch_size",        type=int, default=64)
    parser.add_argument("--umbral",            type=float, default=0.4)
    parser.add_argument("--reload",            action="store_true", help="Forzar recálculo de embeddings")
    parser.add_argument("--hf_token",          default=None, help="Token de HuggingFace (opcional si ya está en caché)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# PARTE 1: Embeddings de subcategorías
# ---------------------------------------------------------------------------

def cargar_subcategorias(path_dir: str, model: SentenceTransformer) -> dict:
    """
    Lee cada .txt del directorio, concatena sus términos en una frase
    y genera un embedding único por subcategoría.
    """
    subcat_embeddings = {}

    archivos = glob.glob(os.path.join(path_dir, "*.txt"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos .txt en: {path_dir}")

    print(f"Cargando {len(archivos)} subcategorías desde {path_dir}...")
    for file_path in tqdm(archivos, desc="Subcategorías"):
        subcat = os.path.splitext(os.path.basename(file_path))[0]

        with open(file_path, "r", encoding="utf-8") as f:
            terms = [line.strip() for line in f if line.strip()]

        large_phrase = " ".join(terms)
        embedding = model.encode(large_phrase, convert_to_tensor=True, show_progress_bar=False)
        subcat_embeddings[subcat] = embedding

    return subcat_embeddings


# ---------------------------------------------------------------------------
# PARTE 2: Embeddings de chunks
# ---------------------------------------------------------------------------

def obtener_embeddings_chunks(
    chunks_df: pd.DataFrame,
    model: SentenceTransformer,
    batch_size: int = 64,
    save_path: str = "chunk_embeddings.npy",
    reload: bool = False,
) -> np.ndarray:
    """
    Calcula o carga embeddings de los chunks.
    Si reload=False y el archivo existe, lo carga directamente.
    """
    if not reload and os.path.exists(save_path):
        print(f"Embeddings encontrados en {save_path}, cargando...")
        embeddings = np.load(save_path)
        print(f"  Forma: {embeddings.shape}")
        return embeddings

    print(f"Calculando embeddings para {len(chunks_df)} chunks...")
    textos = chunks_df["texto_chunk"].tolist()

    chunk_embeddings = model.encode(
        textos,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
        prompt_name="STS",
    )

    embeddings_np = chunk_embeddings.cpu().numpy()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    np.save(save_path, embeddings_np)
    print(f"Embeddings guardados en {save_path}")

    return embeddings_np


# ---------------------------------------------------------------------------
# PARTE 3: Similitudes coseno
# ---------------------------------------------------------------------------

def calcular_similitudes_chunks(
    chunks_df: pd.DataFrame,
    chunk_embeddings: np.ndarray,
    subcat_embeddings: dict,
) -> pd.DataFrame:
    """
    Calcula similitudes coseno entre chunks y subcategorías.
    Devuelve DataFrame con columnas: chunk_id, id_doc, texto_chunk, <subcats...>
    """
    chunk_tensor   = torch.tensor(chunk_embeddings)
    subcats        = list(subcat_embeddings.keys())
    subcat_matrix  = torch.stack(list(subcat_embeddings.values()))

    print("Calculando matriz de similitudes...")
    sim_matrix = util.cos_sim(chunk_tensor, subcat_matrix).cpu().numpy()

    sim_df = pd.DataFrame(sim_matrix, columns=subcats)
    sim_df.insert(0, "chunk_id",    chunks_df["chunk_id"].values)
    sim_df.insert(1, "id_doc",      chunks_df["id_doc"].values)
    sim_df.insert(2, "texto_chunk", chunks_df["texto_chunk"].values)

    return sim_df


# ---------------------------------------------------------------------------
# PARTE 4: Asignación de categorías
# ---------------------------------------------------------------------------

def asignar_categorias(df: pd.DataFrame, umbral: float = 0.30) -> pd.DataFrame:
    """
    Añade columna 'categorias_detectadas' con lista de (categoria, score)
    para cada chunk que supere el umbral.
    """
    subcat_cols = [c for c in df.columns if c not in ("chunk_id", "id_doc", "texto_chunk")]

    def _asignar(fila):
        cats = [(col, fila[col]) for col in subcat_cols if fila[col] >= umbral]
        return cats if cats else [("ninguna", 0)]

    df = df.copy()
    df["categorias_detectadas"] = df.apply(_asignar, axis=1)
    return df


# ---------------------------------------------------------------------------
# PARTE 5: Estadísticos y figura
# ---------------------------------------------------------------------------

def generar_figura_umbral(sim_df: pd.DataFrame, subcat_cols: list, figures_dir: str):
    """
    Gráfico del % de chunks con al menos una asignación según umbral.
    """
    os.makedirs(figures_dir, exist_ok=True)
    sim_temp  = sim_df[subcat_cols].astype(float)
    umbrales  = np.arange(0, 1, 0.01)
    n_total   = len(sim_df)

    pct_asignados = [
        (sim_temp >= u).any(axis=1).sum() / n_total * 100
        for u in umbrales
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(umbrales, pct_asignados, marker="o", markersize=3)
    ax.set_xlabel("Umbral")
    ax.set_ylabel("% de fragmentos con al menos una asignación")
    ax.set_title("Distribución fragmentos elegidos por umbral")
    ax.grid(True)
    fig.tight_layout()

    out_path = os.path.join(figures_dir, "dist_umbral.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Figura guardada en {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Directorios de salida
    results_dir = args.results_dir
    figures_dir = args.figures_dir or os.path.join(results_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)

    # Login HuggingFace (opcional)
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    # ── Modelo ──────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*50}")
    print(f"Dispositivo: {device}")
    print(f"Modelo:      {args.model_id}")
    print(f"{'='*50}\n")

    model = SentenceTransformer(args.model_id).to(device=device)
    print(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Max seq length:        {model.max_seq_length}\n")

    # ── Datos ───────────────────────────────────────────────────────────────
    print(f"Cargando chunks desde {args.chunks_path}...")
    chunks_df = pd.read_parquet(args.chunks_path)
    print(f"  Columnas: {list(chunks_df.columns)}")
    print(f"  Filas:    {len(chunks_df):,}\n")

    # ── Paso 1: Subcategorías ────────────────────────────────────────────────
    subcat_embeddings = cargar_subcategorias(args.tesauro_dir, model)
    print(f"  {len(subcat_embeddings)} subcategorías cargadas.\n")

    # ── Paso 2: Embeddings de chunks ─────────────────────────────────────────
    chunk_embeddings = obtener_embeddings_chunks(
        chunks_df,
        model,
        batch_size=args.batch_size,
        save_path=args.embeddings_path,
        reload=args.reload,
    )

    # ── Paso 3: Similitudes ───────────────────────────────────────────────────
    sim_df = calcular_similitudes_chunks(chunks_df, chunk_embeddings, subcat_embeddings)

    sim_path = os.path.join(results_dir, "similitudes_chunks.xlsx")
    sim_df.to_excel(sim_path, index=False, engine="openpyxl")
    print(f"Similitudes guardadas en {sim_path}\n")

    # ── Paso 4: Asignación de categorías ─────────────────────────────────────
    resultado = asignar_categorias(sim_df, umbral=args.umbral)

    resultado_ciencia = resultado[
        resultado["categorias_detectadas"].apply(lambda x: x[0][0] != "ninguna")
    ]

    ciencia_path = os.path.join(results_dir, "ciencia_chunks.xlsx")
    resultado_ciencia.to_excel(ciencia_path, index=False, engine="openpyxl")
    print(f"Chunks de ciencia guardados en {ciencia_path}")
    print(f"  Chunks con categoría: {len(resultado_ciencia):,} / {len(resultado):,}")
    print(f"  Docs  con categoría:  {resultado_ciencia['id_doc'].nunique():,}\n")

    # ── Paso 5: Estadísticos ──────────────────────────────────────────────────
    subcat_cols = [c for c in sim_df.columns if c not in ("chunk_id", "id_doc", "texto_chunk")]
    sim_temp    = sim_df[subcat_cols].astype(float)

    asignaciones = (sim_temp >= args.umbral).sum().sort_values(ascending=False)
    print("Asignaciones por subcategoría:")
    print(asignaciones.to_string())
    print()
    print("Descriptivos de similitudes:")
    print(sim_temp.describe().T.to_string())

    generar_figura_umbral(sim_df, subcat_cols, figures_dir)

    print("\n✓ Proceso completado.\n")


if __name__ == "__main__":
    main()