# =============================================================================
# EXTRACCIÓN DE FRECUENCIAS POS CON STANZA
# =============================================================================

import ast
import os
from collections import Counter

import pandas as pd
import stanza
from tqdm import tqdm

# =============================================================================
# 1. RUTAS Y PARÁMETROS — AJUSTAR SEGÚN EL ENTORNO
# =============================================================================

PATH_CHUNKS   = r"/home/mst-kmgomezm/hhdd/data/results/chunks_etiquetados_binario.xlsx"  # <<< cambiar
DIR_RESULTS   = r"/home/mst-kmgomezm/hhdd/data/results"                                     # <<< cambiar

USE_GPU = True   # False si no hay GPU disponible

# =============================================================================
# 2. CARGA DE DATOS
# =============================================================================

chunks_etiquetados = pd.read_excel(PATH_CHUNKS)
print(f"Chunks cargados: {len(chunks_etiquetados):,} | Columnas: {chunks_etiquetados.columns.tolist()}")

df_filtrado = chunks_etiquetados[
    ["chunk_id", "id_doc", "texto_chunk", "categorias_detectadas", "etiqueta_ciencia"]
].copy()

print(f"DataFrame filtrado: {len(df_filtrado):,} filas | Columnas: {df_filtrado.columns.tolist()}")
# =============================================================================
# 3. INICIALIZAR PIPELINE STANZA
# =============================================================================

nlp_stanza = stanza.Pipeline(
    lang="es",
    processors="tokenize,pos,lemma",
    use_gpu=USE_GPU,
)
print(f"Pipeline Stanza cargado | GPU: {USE_GPU}")

# =============================================================================
# 4. FUNCIONES
# =============================================================================

def extraer_pos_frecuencias_stanza(textos: list) -> tuple[list, list, list]:
    """
    Procesa una lista de textos con Stanza y devuelve tres listas de diccionarios:
      - verbos_frec[i]:      {lema: frecuencia}
      - adjetivos_frec[i]:   {lema: frecuencia}
      - sustantivos_frec[i]: {lema: frecuencia}
    """
    verbos_frec, adjetivos_frec, sustantivos_frec = [], [], []

    for texto in tqdm(textos, desc="Procesando con Stanza"):
        try:
            if not isinstance(texto, str) or not texto.strip():
                verbos_frec.append({})
                adjetivos_frec.append({})
                sustantivos_frec.append({})
                continue

            doc = nlp_stanza(texto)
            verbos, adjetivos, sustantivos = [], [], []

            for sent in doc.sentences:
                for w in sent.words:
                    if not w.lemma.isalpha():
                        continue
                    if w.upos == "VERB":
                        verbos.append(w.lemma.lower())
                    elif w.upos == "ADJ":
                        adjetivos.append(w.lemma.lower())
                    elif w.upos == "NOUN":
                        sustantivos.append(w.lemma.lower())

            verbos_frec.append(dict(Counter(verbos)))
            adjetivos_frec.append(dict(Counter(adjetivos)))
            sustantivos_frec.append(dict(Counter(sustantivos)))

        except Exception as e:
            print(f"Error procesando texto: {e}")
            verbos_frec.append({})
            adjetivos_frec.append({})
            sustantivos_frec.append({})

    return verbos_frec, adjetivos_frec, sustantivos_frec


def asegurar_dict(x) -> dict:
    """Convierte a dict si el valor es str serializado o ya es dict."""
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return {}
    return {}


def construir_tabla_frecuencias(
    df: pd.DataFrame,
    columna_diccionarios: str,
    columna_ciencia: str = "etiqueta_ciencia",
) -> pd.DataFrame:
    """
    Construye una tabla con frecuencia total, frecuencia en textos de ciencia
    y proporción de ciencia, para cada lema presente en `columna_diccionarios`.

    Returns:
        DataFrame con columnas: palabra, frecuencia_total,
        frecuencia_ciencia, proporcion_ciencia.
    """
    freq_total   = Counter()
    freq_ciencia = Counter()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        dic = asegurar_dict(row[columna_diccionarios])
        if not dic:
            continue
        freq_total.update(dic)
        if row[columna_ciencia] == 1:
            freq_ciencia.update(dic)

    data = [
        {
            "palabra":            palabra,
            "frecuencia_total":   f_total,
            "frecuencia_ciencia": freq_ciencia.get(palabra, 0),
            "proporcion_ciencia": freq_ciencia.get(palabra, 0) / f_total if f_total > 0 else 0,
        }
        for palabra, f_total in freq_total.items()
    ]

    df_resultado = pd.DataFrame(data)

    if df_resultado.empty:
        print("Advertencia: no se generaron frecuencias.")
        return df_resultado

    return df_resultado.sort_values("frecuencia_total", ascending=False).reset_index(drop=True)

# =============================================================================
# 5. PIPELINE — EXTRACCIÓN DE POS
# =============================================================================

textos = df_filtrado["texto_chunk"].tolist()
verbos_frec, adjetivos_frec, sustantivos_frec = extraer_pos_frecuencias_stanza(textos)

# Crear DataFrames con frecuencias por categoría gramatical
df_verbos      = df_filtrado.copy()
df_adjetivos   = df_filtrado.copy()
df_sustantivos = df_filtrado.copy()

df_verbos["verbos_lemas_frecuencias"]           = verbos_frec
df_adjetivos["adjetivos_lemas_frecuencias"]     = adjetivos_frec
df_sustantivos["sustantivos_lemas_frecuencias"] = sustantivos_frec

# Guardar intermedios
os.makedirs(DIR_RESULTS, exist_ok=True)

df_verbos.to_excel(     os.path.join(DIR_RESULTS, "verbos_stanza.xlsx"),      index=False)
df_adjetivos.to_excel(  os.path.join(DIR_RESULTS, "adjetivos_stanza.xlsx"),   index=False)
df_sustantivos.to_excel(os.path.join(DIR_RESULTS, "sustantivos_stanza.xlsx"), index=False)
print("Archivos intermedios guardados.")

# =============================================================================
# 6. PIPELINE — ANÁLISIS DE FRECUENCIAS
# =============================================================================

tabla_verbos = construir_tabla_frecuencias(
    df_verbos, columna_diccionarios="verbos_lemas_frecuencias"
)
tabla_adjetivos = construir_tabla_frecuencias(
    df_adjetivos, columna_diccionarios="adjetivos_lemas_frecuencias"
)
tabla_sustantivos = construir_tabla_frecuencias(
    df_sustantivos, columna_diccionarios="sustantivos_lemas_frecuencias"
)

# Guardar resultados finales en un único Excel con tres hojas
path_out = os.path.join(DIR_RESULTS, "frecuencias_pos_global_vs_ciencia.xlsx")

with pd.ExcelWriter(path_out) as writer:
    tabla_verbos.to_excel(     writer, sheet_name="verbos",      index=False)
    tabla_adjetivos.to_excel(  writer, sheet_name="adjetivos",   index=False)
    tabla_sustantivos.to_excel(writer, sheet_name="sustantivos", index=False)

print(f"Resultados guardados en '{path_out}'")