# =============================================================================
# analisis_estadistico.py
# Análisis estadístico riguroso y pruebas de hipótesis para evaluar la
# divergencia entre canciones originales (Test Set) y generadas por el modelo.
# =============================================================================

import os
import glob
import pandas as pd
import numpy as np
import pretty_midi
from tqdm import tqdm
from scipy.stats import ks_2samp, wasserstein_distance, entropy
from scipy.spatial.distance import jensenshannon

import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR         = 'maestro-v3.0.0'
CSV_PATH         = os.path.join(ROOT_DIR, 'maestro-v3.0.0.csv')
GENERATED_DIR    = 'RESULTADOS_TEST_CLUSTER/midis_generados'
OUT_DIR          = 'RESULTADOS_TEST_CLUSTER/analisis_estadistico'

REPORT_PATH      = os.path.join(OUT_DIR, 'reporte_hipotesis_global.txt')
CSV_METRICS_PATH = os.path.join(OUT_DIR, 'metricas_detalladas.csv')

os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LENGTH = 256  # Notas usadas como semilla (a ignorar en la evaluación)

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN DE DATOS MIDI
# ─────────────────────────────────────────────────────────────────────────────
def get_basic_notes_df(midi_file: str) -> pd.DataFrame:
    """Extrae de forma segura Pitch, Step, Duration y Velocity de un MIDI."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_file)
        if not pm.instruments:
            return pd.DataFrame()

        instrument = pm.instruments[0]
        sorted_notes = sorted(instrument.notes, key=lambda n: n.start)

        if not sorted_notes:
            return pd.DataFrame()

        rows = []
        prev_start = sorted_notes[0].start

        for note in sorted_notes:
            rows.append({
                'pitch': note.pitch,
                'step': note.start - prev_start,
                'duration': note.end - note.start,
                'velocity': note.velocity / 127.0
            })
            prev_start = note.start

        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# MOTOR ESTADÍSTICO
# ─────────────────────────────────────────────────────────────────────────────
def calculate_metrics(orig_col: np.ndarray, gen_col: np.ndarray, is_discrete: bool = False):
    """Calcula el conjunto completo de métricas estadísticas para una característica."""
    if len(orig_col) == 0 or len(gen_col) == 0:
        return None

    stats = {
        'mean_orig': float(orig_col.mean()),
        'mean_gen': float(gen_col.mean()),
        'std_orig': float(orig_col.std()),
        'std_gen': float(gen_col.std()),
        'wasserstein': float(wasserstein_distance(orig_col, gen_col))
    }

    # Prueba de Kolmogorov-Smirnov
    ks_stat, p_val = ks_2samp(orig_col, gen_col)
    stats['ks_stat'] = float(ks_stat)
    stats['p_value'] = float(p_val)
    stats['reject_h0'] = bool(p_val < 0.05)

    if is_discrete:
        # Divergencia KL para variables discretas (Pitch)
        bins = np.arange(129)
        eps = 1e-10
        orig_ph, _ = np.histogram(orig_col, bins=bins, density=True)
        gen_ph,  _ = np.histogram(gen_col, bins=bins, density=True)
        orig_ph = (orig_ph + eps) / (orig_ph + eps).sum()
        gen_ph  = (gen_ph + eps) / (gen_ph + eps).sum()

        stats['kl_div'] = float(entropy(orig_ph, gen_ph))
    else:
        stats['kl_div'] = np.nan # KL no aplica directamente a distribuciones continuas aquí

    return stats

# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*60)
    print(" INICIANDO ANÁLISIS ESTADÍSTICO RIGUROSO".center(60))
    print("="*60 + "\n")

    # 1. Cargar metadatos
    df_meta = pd.read_csv(CSV_PATH)
    test_files = df_meta[df_meta['split'] == 'test']['midi_filename'].tolist()

    global_results = []
    processed_count = 0

    for rel_path in tqdm(test_files, desc="Evaluando Hipótesis"):
        orig_midi_path = os.path.join(ROOT_DIR, rel_path)
        song_name = os.path.basename(orig_midi_path).replace('.midi', '').replace('.mid', '')
        gen_midi_path = os.path.join(GENERATED_DIR, f"{song_name}_GENERATED.mid")

        if not os.path.exists(gen_midi_path):
            continue

        # Extraer notas
        orig_df = get_basic_notes_df(orig_midi_path)
        gen_df = get_basic_notes_df(gen_midi_path)

        # Validar longitudes
        if len(orig_df) <= SEQ_LENGTH or len(gen_df) <= SEQ_LENGTH:
            continue

        # Aislar SOLO la parte inferida (después de la semilla)
        o_cont = orig_df.iloc[SEQ_LENGTH:].copy()
        g_cont = gen_df.iloc[SEQ_LENGTH:].copy()

        # Calcular métricas por característica
        stats_p = calculate_metrics(o_cont['pitch'].values, g_cont['pitch'].values, is_discrete=True)
        stats_s = calculate_metrics(o_cont['step'].values, g_cont['step'].values, is_discrete=False)
        stats_d = calculate_metrics(o_cont['duration'].values, g_cont['duration'].values, is_discrete=False)
        stats_v = calculate_metrics(o_cont['velocity'].values, g_cont['velocity'].values, is_discrete=False)

        if not all([stats_p, stats_s, stats_d, stats_v]):
            continue

        row_data = {'song_name': song_name, 'total_generated_notes': len(g_cont)}

        # Aplanar diccionarios
        for feat_name, stats in zip(['pitch', 'step', 'duration', 'velocity'],
                                    [stats_p, stats_s, stats_d, stats_v]):
            for k, v in stats.items():
                row_data[f"{feat_name}_{k}"] = v

        global_results.append(row_data)
        processed_count += 1

    # 2. Consolidar Resultados
    df_results = pd.DataFrame(global_results)
    df_results.to_csv(CSV_METRICS_PATH, index=False)

    print(f"\n Análisis completado para {processed_count} canciones.")
    print(f" CSV guardado en: {CSV_METRICS_PATH}")

    # 3. Generar Reporte de Texto (Promedios Globales)
    if not df_results.empty:
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(" REPORTE ESTADÍSTICO DE HIPÓTESIS - EVALUACIÓN TEST SET\n")
            f.write("="*70 + "\n\n")

            f.write("HIPÓTESIS NULA (H0): La secuencia generada y la original provienen de la misma distribución subyacente.\n")
            f.write("NIVEL DE SIGNIFICANCIA: alpha = 0.05\n\n")

            features = ['pitch', 'step', 'duration', 'velocity']

            for feat in features:
                f.write(f"--- ANÁLISIS DE LA DIMENSIÓN: {feat.upper()} ---\n")

                # Promedios
                mean_wass = df_results[f"{feat}_wasserstein"].mean()
                mean_ks   = df_results[f"{feat}_ks_stat"].mean()

                # KL solo para Pitch
                if feat == 'pitch':
                    mean_kl = df_results[f"pitch_kl_div"].mean()
                    f.write(f"  KL Divergencia Promedio : {mean_kl:.4f}\n")

                f.write(f"  Dist. Wasserstein Prom. : {mean_wass:.4f}\n")
                f.write(f"  Estadístico KS Promedio : {mean_ks:.4f}\n")

                # Decisión sobre H0 (Porcentaje de rechazo)
                reject_rate = df_results[f"{feat}_reject_h0"].mean() * 100
                f.write(f"  Rechazo de H0           : En el {reject_rate:.1f}% de las canciones evaluadas.\n")

                # Comportamiento poblacional (Medias y Varianzas globales)
                mo = df_results[f"{feat}_mean_orig"].mean()
                mg = df_results[f"{feat}_mean_gen"].mean()
                so = df_results[f"{feat}_std_orig"].mean()
                sg = df_results[f"{feat}_std_gen"].mean()

                f.write(f"  Media Global -> Orig: {mo:.4f} | Gen: {mg:.4f}\n")
                f.write(f"  Desv. Estand -> Orig: {so:.4f} | Gen: {sg:.4f}\n\n")

        print(f" Reporte global guardado en: {REPORT_PATH}")

        # Mostrar un resumen rápido en consola
        with open(REPORT_PATH, 'r', encoding='utf-8') as f:
            print(f.read())