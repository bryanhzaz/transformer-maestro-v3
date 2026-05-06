# =============================================================================
# entrenamientoContinuacion.py
# Continúa el entrenamiento del Transformer Musical desde el checkpoint
# guardado por entrenamientoBase.py
#
# Uso:
#   python entrenamientoContinuacion.py
#
# Requiere en el mismo directorio de trabajo:
#   · RESULTADOS_V3_EXTENDED/mejor_modelo.keras  (o modelo_final.keras)
#   · RESULTADOS_V3_EXTENDED/log_entrenamiento.csv
#   · maestro-v3.0.0/  (datos MIDI originales)
# =============================================================================

import os
import math
import glob
import collections
import warnings
import numpy as np
import pandas as pd
import pretty_midi

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                              mean_squared_error, mean_absolute_error, r2_score)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from scipy.stats import (entropy, ks_2samp, mannwhitneyu,
                          wasserstein_distance, pearsonr)
from scipy.spatial.distance import jensenshannon

# ─────────────────────────────────────────────────────────────────────────────
# SEMILLAS
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# HIPERPARÁMETROS  (deben coincidir exactamente con el entrenamiento original)
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LENGTH    = 256
VOCAB_SIZE    = 128
BATCH_SIZE    = 64
D_MODEL       = 256
NUM_HEADS     = 8
FF_DIM        = 1024
NUM_LAYERS    = 4
DROPOUT_RATE  = 0.10
WARMUP_STEPS  = 4000
MAX_CHORD     = 10.0
MAX_TEMPO     = 240.0
NUM_SONGS     = 10
NUM_PRED      = 128
TEMPERATURE   = 1.0

IDX_PITCH   = 0
IDX_STEP    = 1
IDX_DUR     = 2
IDX_VEL     = 3
IDX_SUSTAIN = 4
IDX_CHORD   = 5
IDX_TEMPO   = 6
IDX_BEAT    = 7
N_FEATURES  = 8

FEAT_COLS = ['pitch', 'step', 'duration', 'velocity',
             'sustain', 'chord_size', 'tempo', 'beat_position']

SCALE = np.array(
    [float(VOCAB_SIZE), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float32
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LA CONTINUACIÓN
# ─────────────────────────────────────────────────────────────────────────────
EXTRA_EPOCHS    = 50       # épocas adicionales de entrenamiento
PATIENCE        = 15       # paciencia del EarlyStopping
NEW_LR_OVERRIDE = None     # None = scheduler original | float = tasa fija ej: 1e-4

ROOT_DIR    = 'maestro-v3.0.0'
OUTPUT_DIR  = 'MIDI_Generado_V3_EXTENDED'
RESULTS_DIR = 'RESULTADOS_V3_EXTENDED'

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CAPA: POSITIONAL ENCODING SINUSOIDAL
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len: int, d_model: int, **kwargs):
        # Keras inyecta 'trainable' desde la config guardada al deserializar,
        # lo quitamos antes del super().__init__() donde ya lo fijamos nosotros.
        kwargs.pop('trainable', None)
        super().__init__(trainable=False, **kwargs)
        self.seq_len = seq_len
        self.d_model = d_model

        pos    = np.arange(seq_len)[:, np.newaxis].astype(np.float32)
        dims   = np.arange(d_model)[np.newaxis,   :].astype(np.float32)
        angles = pos / np.power(10000.0, (2 * (dims // 2)) / float(d_model))
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self._pe = tf.constant(angles[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        return x + self._pe[:, :tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'seq_len': self.seq_len, 'd_model': self.d_model})
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# TASA DE APRENDIZAJE 
# ─────────────────────────────────────────────────────────────────────────────
class TransformerLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model: int, warmup_steps: int = 4000):
        super().__init__()
        self.d_model_val      = float(d_model)
        self.warmup_steps_val = float(warmup_steps)

    def __call__(self, step):
        step  = tf.cast(step, tf.float32) + 1.0
        d_inv = 1.0 / tf.math.sqrt(tf.constant(self.d_model_val))
        arg1  = tf.math.rsqrt(step)
        arg2  = step * (self.warmup_steps_val ** -1.5)
        return d_inv * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {'d_model': self.d_model_val, 'warmup_steps': self.warmup_steps_val}


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN DE PÉRDIDA
# ─────────────────────────────────────────────────────────────────────────────
def mse_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    mse      = tf.square(y_true - y_pred)
    pressure = 10.0 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + pressure)


# ─────────────────────────────────────────────────────────────────────────────
# ARQUITECTURA DEL TRANSFORMER EXTENDIDO (idéntica al original en entrenamientoBase.py)
# ─────────────────────────────────────────────────────────────────────────────
def build_transformer_extended(seq_len:      int   = SEQ_LENGTH,
                               vocab_size:   int   = VOCAB_SIZE,
                               n_features:   int   = N_FEATURES,
                               d_model:      int   = D_MODEL,
                                num_heads:    int   = NUM_HEADS,
                                ff_dim:       int   = FF_DIM,
                                num_layers:   int   = NUM_LAYERS,
                                dropout_rate: float = DROPOUT_RATE
                                ) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(seq_len, n_features), name='feature_sequence')

    pitch_float = inputs[:, :, IDX_PITCH]
    cont_feats  = tf.keras.layers.Lambda(
        lambda x: x[:, :, 1:], name='continuous_features'
    )(inputs)

    _vs = vocab_size
    pitch_int = tf.keras.layers.Lambda(
        lambda x: tf.cast(
            tf.clip_by_value(x * float(_vs), 0.0, float(_vs - 1)),
            tf.int32
        ),
        name='pitch_to_int'
    )(pitch_float)

    emb_dim  = d_model // 2
    cont_dim = d_model - emb_dim

    pitch_emb = tf.keras.layers.Embedding(
        vocab_size, emb_dim, name='pitch_embedding'
    )(pitch_int)

    cont_proj = tf.keras.layers.Dense(
        cont_dim, use_bias=False, name='continuous_proj'
    )(cont_feats)

    x = tf.keras.layers.Concatenate(axis=-1, name='feature_concat')(
        [pitch_emb, cont_proj]
    )
    x = tf.keras.layers.Dense(d_model, use_bias=False, name='input_proj')(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='input_norm')(x)
    x = SinusoidalPositionalEncoding(seq_len, d_model, name='pos_enc')(x)
    x = tf.keras.layers.Dropout(dropout_rate, name='input_drop')(x)

    for i in range(num_layers):
        x_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f'ln_attn_{i}'
        )(x)
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim   = d_model // num_heads,
            dropout   = dropout_rate,
            name      = f'mha_{i}'
        )(x_norm, x_norm, use_causal_mask=True)
        attn = tf.keras.layers.Dropout(dropout_rate, name=f'attn_drop_{i}')(attn)
        x    = tf.keras.layers.Add(name=f'attn_add_{i}')([x, attn])

        x_norm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f'ln_ffn_{i}'
        )(x)
        ffn = tf.keras.layers.Dense(ff_dim, activation='gelu', name=f'ffn1_{i}')(x_norm2)
        ffn = tf.keras.layers.Dropout(dropout_rate, name=f'ffn_drop_{i}')(ffn)
        ffn = tf.keras.layers.Dense(d_model, name=f'ffn2_{i}')(ffn)
        x   = tf.keras.layers.Add(name=f'ffn_add_{i}')([x, ffn])

    x      = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='final_norm')(x)
    x_last = x[:, -1, :]

    out_pitch    = tf.keras.layers.Dense(vocab_size,              name='pitch')(x_last)
    out_step     = tf.keras.layers.Dense(1,                       name='step')(x_last)
    out_duration = tf.keras.layers.Dense(1,                       name='duration')(x_last)
    out_velocity = tf.keras.layers.Dense(1, activation='sigmoid', name='velocity')(x_last)

    return tf.keras.Model(
        inputs,
        {
            'pitch':    out_pitch,
            'step':     out_step,
            'duration': out_duration,
            'velocity': out_velocity,
        },
        name='MusicTransformer_EXTENDED'
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN EXTENDIDA DE CARACTERÍSTICAS MIDI
# ─────────────────────────────────────────────────────────────────────────────
def _sustain_states(control_changes, times: np.ndarray) -> np.ndarray:
    cc64 = sorted(
        [(cc.time, 1.0 if cc.value >= 64 else 0.0)
         for cc in control_changes if cc.number == 64],
        key=lambda x: x[0]
    )
    if not cc64:
        return np.zeros(len(times), dtype=np.float32)
    cc_times  = np.array([c[0] for c in cc64], dtype=np.float32)
    cc_states = np.array([c[1] for c in cc64], dtype=np.float32)
    result = np.zeros(len(times), dtype=np.float32)
    for i, t in enumerate(times):
        idx = np.searchsorted(cc_times, t, side='right') - 1
        if idx >= 0:
            result[i] = cc_states[idx]
    return result


def _chord_sizes(sorted_notes) -> np.ndarray:
    starts = np.array([n.start for n in sorted_notes], dtype=np.float32)
    ends   = np.array([n.end   for n in sorted_notes], dtype=np.float32)
    sizes  = np.array([
        int(np.sum((starts <= n.start) & (ends > n.start)))
        for n in sorted_notes
    ], dtype=np.float32)
    return sizes


def midi_to_notes_extended(midi_file: str) -> pd.DataFrame:
    pm         = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    sorted_notes = sorted(instrument.notes, key=lambda n: n.start)
    if not sorted_notes:
        return pd.DataFrame()

    start_times = np.array([n.start for n in sorted_notes], dtype=np.float32)
    sustain_arr = _sustain_states(instrument.control_changes, start_times)
    chord_arr   = _chord_sizes(sorted_notes)

    tempo_times, tempos = pm.get_tempo_changes()
    if len(tempos) == 0:
        tempo_arr = np.full(len(sorted_notes), 120.0, dtype=np.float32)
    else:
        idx_arr   = np.searchsorted(tempo_times, start_times, side='right') - 1
        idx_arr   = np.clip(idx_arr, 0, len(tempos) - 1)
        tempo_arr = tempos[idx_arr].astype(np.float32)

    beats = pm.get_beats()
    if len(beats) < 2:
        beat_pos_arr = np.zeros(len(sorted_notes), dtype=np.float32)
    else:
        beat_idx_arr = np.searchsorted(beats, start_times, side='right') - 1
        beat_idx_arr = np.clip(beat_idx_arr, 0, len(beats) - 2)
        b_starts     = beats[beat_idx_arr]
        b_ends       = beats[np.minimum(beat_idx_arr + 1, len(beats) - 1)]
        denom        = b_ends - b_starts + 1e-8
        beat_pos_arr = np.clip((start_times - b_starts) / denom, 0.0, 1.0)

    prev_start = sorted_notes[0].start
    rows = collections.defaultdict(list)
    for i, note in enumerate(sorted_notes):
        rows['pitch'].append(note.pitch)
        rows['start'].append(note.start)
        rows['end'].append(note.end)
        rows['step'].append(note.start - prev_start)
        rows['duration'].append(note.end - note.start)
        rows['velocity'].append(note.velocity / 127.0)
        rows['sustain'].append(float(sustain_arr[i]))
        rows['chord_size'].append(min(chord_arr[i] / MAX_CHORD, 1.0))
        rows['tempo'].append(min(tempo_arr[i] / MAX_TEMPO, 1.0))
        rows['beat_position'].append(float(beat_pos_arr[i]))
        prev_start = note.start

    return pd.DataFrame({k: np.array(v) for k, v in rows.items()})


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE DE DATASET
# ─────────────────────────────────────────────────────────────────────────────
def create_sequences_extended(dataset: tf.data.Dataset,
                               seq_length: int) -> tf.data.Dataset:
    win_len   = seq_length + 1
    windows   = dataset.window(win_len, shift=1, stride=1, drop_remainder=True)
    sequences = windows.flat_map(
        lambda w: w.batch(win_len, drop_remainder=True)
    )
    scale_tf = tf.constant(SCALE, dtype=tf.float32)

    def split_labels(seq):
        inputs = seq[:-1]
        label  = seq[-1]
        pitch_label = tf.clip_by_value(
            tf.cast(label[IDX_PITCH], tf.int32), 0, VOCAB_SIZE - 1
        )
        labels = {
            'pitch':    pitch_label,
            'step':     label[IDX_STEP],
            'duration': label[IDX_DUR],
            'velocity': label[IDX_VEL],
        }
        inputs_scaled = inputs / scale_tf
        return inputs_scaled, labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN DE PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────
def predict_next_note_ext(notes: np.ndarray,
                           keras_model: tf.keras.Model,
                           temperature: float = 1.0) -> tuple:
    inputs       = tf.expand_dims(notes, 0)
    preds        = keras_model(inputs, training=False)
    pitch_logits = preds['pitch'][0].numpy() / temperature
    pitch_logits -= np.max(pitch_logits)
    probs         = np.exp(pitch_logits)
    probs        /= probs.sum()
    pitch         = int(np.random.choice(VOCAB_SIZE, p=probs))
    step     = max(float(np.maximum(0.0, preds['step'][0, 0].numpy())),     0.01)
    duration = max(float(np.maximum(0.0, preds['duration'][0, 0].numpy())), 0.05)
    velocity = float(np.clip(preds['velocity'][0, 0].numpy(), 0.0, 1.0))
    return pitch, step, duration, velocity


# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DE MIDI
# ─────────────────────────────────────────────────────────────────────────────
def notes_to_midi_extended(notes_df: pd.DataFrame,
                            out_file: str,
                            instrument_name: str = 'Acoustic Grand Piano'
                            ) -> pretty_midi.PrettyMIDI:
    pm         = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )
    prev_start = 0.0
    for _, row in notes_df.iterrows():
        start    = float(prev_start + row['step'])
        end      = float(start + row['duration'])
        velocity = int(np.clip(round(row.get('velocity', 0.63) * 127), 1, 127))
        instrument.notes.append(pretty_midi.Note(
            velocity=velocity, pitch=int(row['pitch']),
            start=start, end=end
        ))
        prev_start = start
    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS ESTADÍSTICAS
# ─────────────────────────────────────────────────────────────────────────────
def compute_stats_extended(original_df: pd.DataFrame,
                            generated_df: pd.DataFrame,
                            results_file) -> None:
    EPS  = 1e-10
    bins = np.arange(129)

    orig_ph, _ = np.histogram(original_df['pitch'],  bins=bins, density=True)
    gen_ph,  _ = np.histogram(generated_df['pitch'], bins=bins, density=True)
    orig_ph = (orig_ph + EPS) / (orig_ph + EPS).sum()
    gen_ph  = (gen_ph  + EPS) / (gen_ph  + EPS).sum()

    kl_div  = float(entropy(orig_ph, gen_ph))
    js_div  = float(jensenshannon(orig_ph, gen_ph))
    w_pitch = float(wasserstein_distance(original_df['pitch'], generated_df['pitch']))
    ks_p, pv_p = ks_2samp(original_df['pitch'], generated_df['pitch'])

    orig_pc = np.bincount(original_df['pitch'].astype(int) % 12, minlength=12).astype(float)
    gen_pc  = np.bincount(generated_df['pitch'].astype(int) % 12, minlength=12).astype(float)
    orig_pc /= orig_pc.sum() + EPS
    gen_pc  /= gen_pc.sum()  + EPS
    js_pc    = float(jensenshannon(orig_pc, gen_pc))

    ks_s, pv_s   = ks_2samp(original_df['step'], generated_df['step'])
    mw_s, mwp_s  = mannwhitneyu(original_df['step'], generated_df['step'], alternative='two-sided')
    w_step       = float(wasserstein_distance(original_df['step'], generated_df['step']))
    min_len_s    = min(len(original_df), len(generated_df))
    pr_s, _      = pearsonr(
        original_df['step'].values[:min_len_s],
        generated_df['step'].values[:min_len_s]
    ) if min_len_s >= 2 else (float('nan'), None)

    ks_d, pv_d   = ks_2samp(original_df['duration'], generated_df['duration'])
    mw_d, mwp_d  = mannwhitneyu(original_df['duration'], generated_df['duration'], alternative='two-sided')
    w_dur        = float(wasserstein_distance(original_df['duration'], generated_df['duration']))
    pr_d, _      = pearsonr(
        original_df['duration'].values[:min_len_s],
        generated_df['duration'].values[:min_len_s]
    ) if min_len_s >= 2 else (float('nan'), None)

    vel_stats = ""
    if 'velocity' in original_df.columns and 'velocity' in generated_df.columns:
        ks_v, pv_v = ks_2samp(original_df['velocity'], generated_df['velocity'])
        w_vel      = float(wasserstein_distance(original_df['velocity'], generated_df['velocity']))
        vel_stats  = (
            f"[VELOCITY]\n"
            f"  KS  Estadístico    : {ks_v:.4f}   p-valor = {pv_v:.4e}\n"
            f"  Wasserstein Dist.  : {w_vel:.6f}\n"
            f"  media orig={original_df['velocity'].mean():.4f} "
            f"gen={generated_df['velocity'].mean():.4f}\n\n"
        )

    results_file.write("\n── PRUEBAS ESTADÍSTICAS: ORIGINAL vs GENERADA ──\n\n")
    results_file.write("[PITCH]\n")
    results_file.write(f"  KL  Divergencia    : {kl_div:.6f}\n")
    results_file.write(f"  JS  Divergencia    : {js_div:.6f}  (0=idénticas)\n")
    results_file.write(f"  Wasserstein Dist.  : {w_pitch:.4f}\n")
    results_file.write(f"  KS  Estadístico    : {ks_p:.4f}   p-valor = {pv_p:.4e}\n")
    results_file.write(f"  JS  Pitch Class    : {js_pc:.6f}  (similitud tonal)\n\n")
    results_file.write("[STEP]\n")
    results_file.write(f"  KS  Estadístico    : {ks_s:.4f}   p-valor = {pv_s:.4e}\n")
    results_file.write(f"  Mann-Whitney U     : {mw_s:.2f}  p-valor = {mwp_s:.4e}\n")
    results_file.write(f"  Wasserstein Dist.  : {w_step:.6f}\n")
    results_file.write(f"  Pearson r          : {pr_s:.4f}\n\n")
    results_file.write("[DURATION]\n")
    results_file.write(f"  KS  Estadístico    : {ks_d:.4f}   p-valor = {pv_d:.4e}\n")
    results_file.write(f"  Mann-Whitney U     : {mw_d:.2f}  p-valor = {mwp_d:.4e}\n")
    results_file.write(f"  Wasserstein Dist.  : {w_dur:.6f}\n")
    results_file.write(f"  Pearson r          : {pr_d:.4f}\n\n")
    if vel_stats:
        results_file.write(vel_stats)

    for feat in ['pitch', 'step', 'duration']:
        if feat in original_df.columns and feat in generated_df.columns:
            o, g = original_df[feat], generated_df[feat]
            results_file.write(
                f"[{feat.upper():8s}] "
                f"media orig={o.mean():.4f} gen={g.mean():.4f} | "
                f"std orig={o.std():.4f} gen={g.std():.4f} | "
                f"med orig={o.median():.4f} gen={g.median():.4f}\n"
            )


def plot_comparison_ext(original_df: pd.DataFrame,
                        generated_df: pd.DataFrame,
                        tag: str = '') -> None:
    feats    = ['pitch', 'step', 'duration', 'velocity']
    colors_o = ['steelblue', 'seagreen', 'coral',      'mediumpurple']
    colors_g = ['orange',    'tomato',   'dodgerblue',  'gold']
    fig, axes = plt.subplots(1, len(feats), figsize=(20, 4))
    for ax, feat, co, cg in zip(axes, feats, colors_o, colors_g):
        if feat in original_df.columns:
            ax.hist(original_df[feat],  bins=40, alpha=0.6, color=co, label='Original', density=True)
        if feat in generated_df.columns:
            ax.hist(generated_df[feat], bins=40, alpha=0.6, color=cg, label='Generada', density=True)
        ax.set_title(feat.capitalize())
        ax.legend()
    plt.suptitle(f'Distribuciones Original vs Generada {tag}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'comparacion{tag}.png'), dpi=120)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDAD: leer épocas ya entrenadas desde el CSV log
# ─────────────────────────────────────────────────────────────────────────────
def get_epochs_trained(log_path: str) -> int:
    if not os.path.exists(log_path):
        print(f"[AVISO] No se encontró el log en {log_path}. Se asumirá initial_epoch=0.")
        return 0
    try:
        df  = pd.read_csv(log_path)
        col = 'epoch' if 'epoch' in df.columns else df.columns[0]
        last_epoch = int(df[col].max()) + 1   # epoch col es 0-based
        print(f"[INFO] Épocas ya entrenadas según log: {last_epoch}")
        return last_epoch
    except Exception as e:
        print(f"[AVISO] No se pudo leer el log ({e}). Se asumirá initial_epoch=0.")
        return 0


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == '__main__':

    # ── 1. Determinar épocas ya corridas ──────────────────────────────────────
    log_path      = os.path.join(RESULTS_DIR, 'log_entrenamiento.csv')
    initial_epoch = get_epochs_trained(log_path)
    total_epochs  = initial_epoch + EXTRA_EPOCHS

    print(f"\n{'='*60}")
    print(f"  CONTINUACIÓN DEL ENTRENAMIENTO")
    print(f"  Épocas previas   : {initial_epoch}")
    print(f"  Épocas extra     : {EXTRA_EPOCHS}")
    print(f"  Total objetivo   : {total_epochs}")
    print(f"{'='*60}\n")

    # ── 2. Localizar checkpoint ───────────────────────────────────────────────
    best_ckpt  = os.path.join(RESULTS_DIR, 'mejor_modelo.keras')
    final_ckpt = os.path.join(RESULTS_DIR, 'modelo_final.keras')

    if os.path.exists(best_ckpt):
        ckpt_path = best_ckpt
        print(f"Cargando pesos desde: {ckpt_path}")
    elif os.path.exists(final_ckpt):
        ckpt_path = final_ckpt
        print(f"Cargando pesos desde: {ckpt_path}")
    else:
        raise FileNotFoundError(
            "No se encontró ningún modelo guardado en:\n"
            f"  {best_ckpt}\n  {final_ckpt}"
        )

    # ── 3. Construir modelo y cargar pesos ────────────────────────────────────
    # Se reconstruye la arquitectura desde cero para evitar todos los problemas
    # de deserialización de capas Lambda y SinusoidalPositionalEncoding.
    # Luego se cargan únicamente los pesos del checkpoint.
    strategy = tf.distribute.MirroredStrategy()
    print(f"Réplicas GPU disponibles: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        model = build_transformer_extended()

        loss_fns = {
            'pitch':    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            'step':     mse_positive_pressure,
            'duration': mse_positive_pressure,
            'velocity': tf.keras.losses.MeanSquaredError(),
        }
        loss_weights = {
            'pitch':    1.0,
            'step':     1.0,
            'duration': 1.0,
            'velocity': 0.5,
        }

        if NEW_LR_OVERRIDE is not None:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate = NEW_LR_OVERRIDE,
                beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipnorm=1.0
            )
            print(f"[INFO] Tasa de aprendizaje fija: {NEW_LR_OVERRIDE}")
        else:
            optimizer = tf.keras.optimizers.Adam(
                TransformerLRSchedule(D_MODEL, WARMUP_STEPS),
                beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipnorm=1.0
            )

        model.compile(
            optimizer    = optimizer,
            loss         = loss_fns,
            loss_weights = loss_weights
        )

        # Cargar solo los pesos (evita todos los problemas de deserialización)
        model.load_weights(ckpt_path)
        print(" Pesos cargados correctamente.")

    model.summary()

    # ── 4. Reconstruir el dataset ─────────────────────────────────────────────
    filenames = glob.glob(os.path.join(ROOT_DIR, '**/*.mid*'), recursive=True)
    print(f"\nArchivos MIDI encontrados: {len(filenames)}")
    if not filenames:
        raise FileNotFoundError(f"No se encontraron archivos en {ROOT_DIR}")

    all_notes = []
    exitosos  = 0
    fallidos  = 0
    total     = len(filenames)

    print(f"\nExtrayendo características de {total} archivos MIDI...")
    for f in filenames:
        try:
            df = midi_to_notes_extended(f)
            if df is not None and len(df) > SEQ_LENGTH:
                all_notes.append(df)
                exitosos += 1
            if exitosos % 100 == 0 and exitosos > 0:
                print(f"  ✓ {exitosos}/{total} archivos procesados")
        except Exception:
            fallidos += 1

    print(f"\n   Exitosos : {exitosos}")
    print(f"   Fallidos : {fallidos}")

    all_notes_df = pd.concat(all_notes, ignore_index=True)
    n_notes      = len(all_notes_df)
    print(f"   Total de eventos : {n_notes:,}")

    feat_arr = np.stack(
        [all_notes_df[col].values for col in FEAT_COLS], axis=1
    ).astype(np.float32)

    notes_ds     = tf.data.Dataset.from_tensor_slices(feat_arr)
    seq_ds       = create_sequences_extended(notes_ds, SEQ_LENGTH)
    dataset_size = n_notes - SEQ_LENGTH
    train_size   = int(0.85 * dataset_size)
    buffer_size  = min(200_000, dataset_size)

    seq_ds = seq_ds.shuffle(buffer_size, seed=SEED, reshuffle_each_iteration=False)

    train_ds = (
        seq_ds.take(train_size)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        seq_ds.skip(train_size)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    print(f"\nDataset listo: ~{train_size:,} secuencias de entrenamiento")

    # ── 5. Callbacks ──────────────────────────────────────────────────────────
    new_best_ckpt = os.path.join(RESULTS_DIR, 'mejor_modelo_cont.keras')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor              = 'val_loss',
            patience             = PATIENCE,
            restore_best_weights = True,
            verbose              = 1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath       = new_best_ckpt,
            monitor        = 'val_loss',
            save_best_only = True,
            verbose        = 1
        ),
        tf.keras.callbacks.CSVLogger(
            log_path,
            append = True    # continúa el log existente sin sobreescribir
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 0.5,
            patience = 5,
            min_lr   = 1e-6,
            verbose  = 1
        ),
    ]

    # ── 6. Continuar entrenamiento ────────────────────────────────────────────
    print(f"\nReanudando desde época {initial_epoch} hasta {total_epochs}...")

    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = total_epochs,
        initial_epoch   = initial_epoch,    #  clave para no repetir épocas
        callbacks       = callbacks
    )

    cont_final = os.path.join(RESULTS_DIR, 'modelo_continuado_final.keras')
    model.save(cont_final)
    print(f"Modelo guardado en: {cont_final}")

    # ── 7. Curva de pérdida completa ──────────────────────────────────────────
    try:
        log_df = pd.read_csv(log_path)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(log_df['epoch'], log_df['loss'],     label='Train Total')
        ax.plot(log_df['epoch'], log_df['val_loss'], label='Val Total')
        for col, lbl, ls in [
            ('pitch_loss',    'Pitch',    '--'),
            ('step_loss',     'Step',     '--'),
            ('duration_loss', 'Duration', '--'),
            ('velocity_loss', 'Velocity', ':'),
        ]:
            if col in log_df:
                ax.plot(log_df['epoch'], log_df[col], linestyle=ls, label=lbl)
        ax.axvline(x=initial_epoch - 1, color='red', linestyle=':',
                   alpha=0.7, label=f'Reanudación (época {initial_epoch})')
        ax.set_xlabel('Época')
        ax.set_ylabel('Pérdida')
        ax.set_title('Historial Completo (Original + Continuación)')
        ax.legend()
        plt.tight_layout()
        curve_path = os.path.join(RESULTS_DIR, 'historial_completo.png')
        plt.savefig(curve_path, dpi=120)
        plt.close()
        print(f"Curva guardada en: {curve_path}")
    except Exception as e:
        print(f"[AVISO] No se pudo graficar: {e}")

    # ── 8. Métricas en validación ─────────────────────────────────────────────
    print("\nCalculando métricas sobre el set de validación...")

    y_true_pitch, y_pred_pitch = [], []
    y_true_step,  y_pred_step  = [], []
    y_true_dur,   y_pred_dur   = [], []
    y_true_vel,   y_pred_vel   = [], []

    for x_batch, y_batch in val_ds.take(150):
        preds = model(x_batch, training=False)
        y_true_pitch.extend(y_batch['pitch'].numpy())
        y_true_step.extend( y_batch['step'].numpy())
        y_true_dur.extend(  y_batch['duration'].numpy())
        y_true_vel.extend(  y_batch['velocity'].numpy())
        y_pred_pitch.extend(np.argmax(preds['pitch'].numpy(), axis=-1))
        y_pred_step.extend( np.maximum(0.0, preds['step'].numpy().flatten()))
        y_pred_dur.extend(  np.maximum(0.0, preds['duration'].numpy().flatten()))
        y_pred_vel.extend(  np.clip(preds['velocity'].numpy().flatten(), 0.0, 1.0))

    y_true_pitch = np.array(y_true_pitch)
    y_pred_pitch = np.array(y_pred_pitch)
    y_true_step  = np.array(y_true_step)
    y_pred_step  = np.array(y_pred_step)
    y_true_dur   = np.array(y_true_dur)
    y_pred_dur   = np.array(y_pred_dur)
    y_true_vel   = np.array(y_true_vel)
    y_pred_vel   = np.array(y_pred_vel)

    acc = accuracy_score(y_true_pitch, y_pred_pitch)
    f1  = f1_score(      y_true_pitch, y_pred_pitch, average='weighted', zero_division=0)
    rec = recall_score(  y_true_pitch, y_pred_pitch, average='weighted', zero_division=0)

    try:
        last_pitch_loss = history.history.get('pitch_loss', [float('inf')])[-1]
        perplexity = math.exp(min(last_pitch_loss, 700))
    except Exception:
        perplexity = float('inf')

    smoother = SmoothingFunction().method1
    bleu_1, bleu_2, bleu_4 = [], [], []
    chunk = 20
    for i in range(0, len(y_true_pitch) - chunk, chunk):
        ref  = [[str(t) for t in y_true_pitch[i:i+chunk]]]
        cand =  [str(t) for t in y_pred_pitch[i:i+chunk]]
        bleu_1.append(sentence_bleu(ref, cand, weights=(1,0,0,0),     smoothing_function=smoother))
        bleu_2.append(sentence_bleu(ref, cand, weights=(0.5,0.5,0,0), smoothing_function=smoother))
        bleu_4.append(sentence_bleu(ref, cand, weights=(0.25,)*4,     smoothing_function=smoother))

    mse_s = mean_squared_error( y_true_step, y_pred_step)
    mae_s = mean_absolute_error(y_true_step, y_pred_step)
    r2_s  = r2_score(           y_true_step, y_pred_step)
    mse_d = mean_squared_error( y_true_dur,  y_pred_dur)
    mae_d = mean_absolute_error(y_true_dur,  y_pred_dur)
    r2_d  = r2_score(           y_true_dur,  y_pred_dur)
    mse_v = mean_squared_error( y_true_vel,  y_pred_vel)
    mae_v = mean_absolute_error(y_true_vel,  y_pred_vel)
    r2_v  = r2_score(           y_true_vel,  y_pred_vel)

    # ── 9. Generación de canciones ────────────────────────────────────────────
    print(f"\nGenerando {NUM_SONGS} canciones (temperatura={TEMPERATURE})...")
    np.random.shuffle(filenames)

    first_generated_df = None
    first_original_df  = None
    pistas_ok          = 0

    for midi_file in filenames:
        if pistas_ok >= NUM_SONGS:
            break
        try:
            raw_ext = midi_to_notes_extended(midi_file)
            if raw_ext is None or len(raw_ext) <= SEQ_LENGTH:
                continue
        except Exception:
            continue

        feat_mat = np.stack(
            [raw_ext[col].values for col in FEAT_COLS], axis=1
        ).astype(np.float32)

        idx = np.random.randint(0, len(feat_mat) - SEQ_LENGTH)
        win = feat_mat[idx:idx+SEQ_LENGTH].copy()
        win[:, IDX_PITCH] /= float(VOCAB_SIZE)

        generated  = []
        prev_start = 0.0

        for _ in range(NUM_PRED):
            pitch, step, duration, velocity = predict_next_note_ext(win, model, TEMPERATURE)
            start = prev_start + step
            end   = start + duration
            generated.append((pitch, step, duration, velocity, start, end))

            new_row = np.array([
                pitch / VOCAB_SIZE,
                step, duration, velocity,
                0.0,
                min(1.0 / MAX_CHORD, 1.0),
                win[-1, IDX_TEMPO],
                0.0
            ], dtype=np.float32)

            win = np.concatenate([win[1:], new_row[np.newaxis, :]], axis=0)
            prev_start = start

        gen_df = pd.DataFrame(
            generated,
            columns=['pitch', 'step', 'duration', 'velocity', 'start', 'end']
        )
        out_path = os.path.join(OUTPUT_DIR, f'cancion_CONT_{pistas_ok+1:02d}.mid')
        notes_to_midi_extended(gen_df, out_path)

        if pistas_ok == 0:
            first_generated_df = gen_df
            first_original_df  = raw_ext

        print(f"  → Pista {pistas_ok+1}/{NUM_SONGS} guardada en {out_path}")
        pistas_ok += 1

    # ── 10. Guardar métricas ──────────────────────────────────────────────────
    metrics_path = os.path.join(RESULTS_DIR, 'metricas_continuacion.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"  MÉTRICAS — Continuación desde época {initial_epoch}\n")
        f.write(f"  Total épocas entrenadas: {total_epochs}\n")
        f.write("=" * 60 + "\n\n")

        f.write("── CLASIFICACIÓN (PITCH) ──\n")
        f.write(f"  Accuracy   : {acc:.4f}\n")
        f.write(f"  F1-Score   : {f1:.4f}\n")
        f.write(f"  Recall     : {rec:.4f}\n")
        f.write(f"  Perplexity : {perplexity:.4f}\n")
        if bleu_1:
            f.write(f"  BLEU-1     : {np.mean(bleu_1):.4f}\n")
            f.write(f"  BLEU-2     : {np.mean(bleu_2):.4f}\n")
            f.write(f"  BLEU-4     : {np.mean(bleu_4):.4f}\n")
        f.write("\n")

        f.write("── REGRESIÓN (STEP) ──\n")
        f.write(f"  MSE : {mse_s:.6f} | MAE : {mae_s:.6f} | R² : {r2_s:.4f}\n\n")
        f.write("── REGRESIÓN (DURATION) ──\n")
        f.write(f"  MSE : {mse_d:.6f} | MAE : {mae_d:.6f} | R² : {r2_d:.4f}\n\n")
        f.write("── REGRESIÓN (VELOCITY) ──\n")
        f.write(f"  MSE : {mse_v:.6f} | MAE : {mae_v:.6f} | R² : {r2_v:.4f}\n\n")

        f.write("── COMPARATIVA CON MÉTRICAS PREVIAS ──\n")
        f.write(f"  Accuracy    prev: 0.5201  →  ahora: {acc:.4f}\n")
        f.write(f"  R² Step     prev: 0.1271  →  ahora: {r2_s:.4f}\n")
        f.write(f"  R² Duration prev: 0.4427  →  ahora: {r2_d:.4f}\n")
        f.write(f"  R² Velocity prev: 0.5642  →  ahora: {r2_v:.4f}\n\n")

        if first_generated_df is not None and first_original_df is not None:
            compute_stats_extended(first_original_df, first_generated_df, f)
            plot_comparison_ext(first_original_df, first_generated_df, tag='_CONT')

    print(f"\n Métricas guardadas en: {metrics_path}")
    print("¡Continuación del entrenamiento completada!")
    print(f"\nArchivos generados:")
    print(f"  Mejor modelo     → {new_best_ckpt}")
    print(f"  Modelo final     → {cont_final}")
    print(f"  Métricas         → {metrics_path}")
    print(f"  Curva de pérdida → {os.path.join(RESULTS_DIR, 'historial_completo.png')}")