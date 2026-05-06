# =============================================================================
# entrenamiento_v3_EXTENDED.py
# Transformer para Generación Musical — MAESTRO v3.0.0
#
# Características de ENTRADA (8 en total):
#   1. pitch          → Embedding(128, d_model//2)    [discreto]
#   2. step           → Dense  [continua, segundos]
#   3. duration       → Dense  [continua, segundos]
#   4. velocity       → Dense  [continua, normalizada 0-1]
#   5. sustain        → Dense  [binaria: pedal sostenido activo]
#   6. chord_size     → Dense  [continua, notas simultáneas norm.]
#   7. tempo          → Dense  [continua, BPM normalizado /240]
#   8. beat_position  → Dense  [continua, posición en el compás 0-1]
#
# Salidas del modelo: pitch, step, duration, velocity
# Salidas: RESULTADOS_V3_EXTENDED / MIDI_Generado_V3_EXTENDED
# =============================================================================

import os
import math
import glob
import collections
import warnings
import numpy as np
import pandas as pd
import pretty_midi
import seaborn as sns

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
# SEMILLAS Y REPRODUCIBILIDAD
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# HIPERPARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LENGTH    = 256
VOCAB_SIZE    = 128
BATCH_SIZE    = 64
D_MODEL       = 256
NUM_HEADS     = 8
FF_DIM        = 1024    # 4 × D_MODEL
NUM_LAYERS    = 4
DROPOUT_RATE  = 0.10
WARMUP_STEPS  = 4000
EPOCHS        = 50
NUM_SONGS     = 10
NUM_PRED      = 128
TEMPERATURE   = 1.0
MAX_CHORD     = 10.0    # normalización de chord_size
MAX_TEMPO     = 240.0   # normalización de tempo (BPM)

# Índices en el vector de características
# [pitch_norm, step, duration, velocity, sustain, chord_size, tempo, beat_pos]
IDX_PITCH    = 0
IDX_STEP     = 1
IDX_DUR      = 2
IDX_VEL      = 3
IDX_SUSTAIN  = 4
IDX_CHORD    = 5
IDX_TEMPO    = 6
IDX_BEAT     = 7
N_FEATURES   = 8

KEY_ORDER_BASE = ['pitch', 'step', 'duration']

# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORIOS
# ─────────────────────────────────────────────────────────────────────────────
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
        """
        Capa para codificar la posición de los tokens en la secuencia.
        Se utiliza la codificación posicional sinusoidal para añadir información sobre la posición de cada token.
        """
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
# EXTRACCIÓN DE CARACTERÍSTICAS MIDI
# ─────────────────────────────────────────────────────────────────────────────
def _sustain_states(control_changes, times: np.ndarray) -> np.ndarray:
    """
    Función que devuelve 1.0 si el pedal de sustain (CC64 >= 64) está activo
    en cada instante de `times`, 0.0 en caso contrario.
    """
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
    """
    Para cada nota, cuenta cuántas notas están sonando simultáneamente
    en el momento de su ataque (incluida ella misma).
    """
    starts = np.array([n.start for n in sorted_notes], dtype=np.float32)
    ends   = np.array([n.end   for n in sorted_notes], dtype=np.float32)
    sizes  = np.array([
        int(np.sum((starts <= n.start) & (ends > n.start)))
        for n in sorted_notes
    ], dtype=np.float32)
    return sizes


def midi_to_notes_extended(midi_file: str) -> pd.DataFrame:
    """
    Extrae las 8 características por nota de un archivo MIDI.
    Retorna DataFrame con columnas:
        pitch, step, duration, velocity,
        sustain, chord_size, tempo, beat_position
    """
    pm         = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]

    # Notas ordenadas por tiempo de ataque
    sorted_notes = sorted(instrument.notes, key=lambda n: n.start)
    if not sorted_notes:
        return pd.DataFrame()

    start_times = np.array([n.start for n in sorted_notes], dtype=np.float32)

    # ── Sustain pedal ─────────────────────────────────────────────────────────
    sustain_arr = _sustain_states(instrument.control_changes, start_times)

    # ── Tamaño de acorde ──────────────────────────────────────────────────────
    chord_arr = _chord_sizes(sorted_notes)

    # ── Tempo en cada ataque ──────────────────────────────────────────────────
    tempo_times, tempos = pm.get_tempo_changes()
    if len(tempos) == 0:
        tempo_arr = np.full(len(sorted_notes), 120.0, dtype=np.float32)
    else:
        idx_arr   = np.searchsorted(tempo_times, start_times, side='right') - 1
        idx_arr   = np.clip(idx_arr, 0, len(tempos) - 1)
        tempo_arr = tempos[idx_arr].astype(np.float32)

    # ── Posición en el tiempo del compás ─────────────────────────────────────
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

    # ── Construir DataFrame ───────────────────────────────────────────────────
    prev_start = sorted_notes[0].start
    rows = collections.defaultdict(list)

    for i, note in enumerate(sorted_notes):
        rows['pitch'].append(note.pitch)
        rows['start'].append(note.start)
        rows['end'].append(note.end)
        rows['step'].append(note.start - prev_start)
        rows['duration'].append(note.end - note.start)
        rows['velocity'].append(note.velocity / 127.0)          # → [0, 1]
        rows['sustain'].append(float(sustain_arr[i]))
        rows['chord_size'].append(
            min(chord_arr[i] / MAX_CHORD, 1.0)                  # → [0, 1]
        )
        rows['tempo'].append(
            min(tempo_arr[i] / MAX_TEMPO, 1.0)                  # → [0, 1]
        )
        rows['beat_position'].append(float(beat_pos_arr[i]))    # ya [0, 1]
        prev_start = note.start

    return pd.DataFrame({k: np.array(v) for k, v in rows.items()})


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICAS PARA 
# ─────────────────────────────────────────────────────────────────────────────
def plot_piano_roll(notes: pd.DataFrame, count: int = 100):
    plt.figure(figsize=(20, 4))
    pp = np.stack([notes['pitch'], notes['pitch']], axis=0)
    ts = np.stack([notes['start'], notes['end']],   axis=0)
    plt.plot(ts[:, :count], pp[:, :count],
             color='steelblue', marker='.', linewidth=0.8)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Pitch')
    plt.title(f'Piano Roll — primeras {count} notas')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'piano_roll_muestra.png'), dpi=120)
    plt.close()


def plot_feature_distributions(df: pd.DataFrame, tag: str = ''):
    feats = ['pitch', 'step', 'duration', 'velocity',
             'sustain', 'chord_size', 'tempo', 'beat_position']
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, feat in zip(axes.flatten(), feats):
        if feat in df.columns:
            ax.hist(df[feat], bins=30, color='steelblue', alpha=0.75)
            ax.set_title(feat)
    plt.suptitle(f'Distribuciones de características{tag}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'distribuciones{tag}.png'), dpi=120)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE DE DATASET
# ─────────────────────────────────────────────────────────────────────────────
# Columnas en el tensor de características (índices definidos arriba)
FEAT_COLS = ['pitch', 'step', 'duration', 'velocity',
             'sustain', 'chord_size', 'tempo', 'beat_position']

# Escala de normalización por columna:
#   pitch /= 128, velocity/sustain/chord/tempo/beat ya normalizados
SCALE = np.array(
    [float(VOCAB_SIZE), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float32
)


def create_sequences_extended(dataset: tf.data.Dataset,
                               seq_length: int) -> tf.data.Dataset:
    """
    Genera pares (entrada, etiquetas) para el modelo extendido.
    Entrada  : (seq_length, 8) — todas las características normalizadas
    Etiquetas: pitch (int), step (float), duration (float), velocity (float)
    """
    win_len   = seq_length + 1
    windows   = dataset.window(win_len, shift=1, stride=1, drop_remainder=True)
    sequences = windows.flat_map(
        lambda w: w.batch(win_len, drop_remainder=True)
    )

    scale_tf = tf.constant(SCALE, dtype=tf.float32)

    def split_labels(seq):
        # seq: (win_len, 8)  — valores SIN normalizar (excepto los ya norm.)
        inputs = seq[:-1]    # (seq_length, 8)
        label  = seq[-1]     # (8,)

        pitch_label = tf.clip_by_value(
            tf.cast(label[IDX_PITCH], tf.int32), 0, VOCAB_SIZE - 1
        )
        labels = {
            'pitch':    pitch_label,
            'step':     label[IDX_STEP],
            'duration': label[IDX_DUR],
            'velocity': label[IDX_VEL],    # ya en [0,1] desde la extracción
        }
        # Normalizar pitch de la entrada (el resto ya está en [0,1])
        inputs_scaled = inputs / scale_tf
        return inputs_scaled, labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN DE PÉRDIDA
# ─────────────────────────────────────────────────────────────────────────────
def mse_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    mse      = tf.square(y_true - y_pred)
    pressure = 10.0 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + pressure)


# ─────────────────────────────────────────────────────────────────────────────
# ARQUITECTURA DEL TRANSFORMER 
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
    """
    Transformer decoder-only con representación de entrada mixta:
      · Pitch (discreto) → Embedding
      · [step, duration, velocity, sustain, chord_size, tempo, beat_pos]
        (continuas) → Dense proyección conjunta
    Los vectores se concatenan y proyectan a d_model.
    Pre-LayerNorm + GELU en FFN + máscara causal.
    4 salidas: pitch, step, duration, velocity.
    """
    inputs = tf.keras.Input(shape=(seq_len, n_features), name='feature_sequence')

    # ── Separación de características ────────────────────────────────────────
    pitch_float = inputs[:, :, IDX_PITCH]      # normalizado (0 – 1)
    # Las 7 características continuas (índices 1-7)
    cont_feats  = tf.keras.layers.Lambda(
        lambda x: x[:, :, 1:], name='continuous_features'
    )(inputs)                                  # (B, T, 7)

    # Recuperar índice de pitch para el Embedding
    # NOTA: En Keras 3 tf.cast no puede operar sobre KerasTensor directamente.
    # Solución: envolver en Lambda para que la operación ocurra dentro de una capa.
    _vs = vocab_size  # capturar en closure
    pitch_int = tf.keras.layers.Lambda(
        lambda x: tf.cast(
            tf.clip_by_value(x * float(_vs), 0.0, float(_vs - 1)),
            tf.int32
        ),
        name='pitch_to_int'
    )(pitch_float)

    # ── Representación de entrada ─────────────────────────────────────────────
    emb_dim  = d_model // 2
    cont_dim = d_model - emb_dim

    pitch_emb = tf.keras.layers.Embedding(
        vocab_size, emb_dim, name='pitch_embedding'
    )(pitch_int)                                              # (B, T, emb_dim)

    cont_proj = tf.keras.layers.Dense(
        cont_dim, use_bias=False, name='continuous_proj'
    )(cont_feats)                                             # (B, T, cont_dim)

    x = tf.keras.layers.Concatenate(axis=-1, name='feature_concat')(
        [pitch_emb, cont_proj]
    )                                                         # (B, T, d_model)
    x = tf.keras.layers.Dense(d_model, use_bias=False, name='input_proj')(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='input_norm')(x)

    x = SinusoidalPositionalEncoding(seq_len, d_model, name='pos_enc')(x)
    x = tf.keras.layers.Dropout(dropout_rate, name='input_drop')(x)

    # ── Bloques Transformer (Pre-LN) ─────────────────────────────────────────
    for i in range(num_layers):
        # Atención causal
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

        # FFN con GELU
        x_norm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f'ln_ffn_{i}'
        )(x)
        ffn = tf.keras.layers.Dense(ff_dim, activation='gelu', name=f'ffn1_{i}')(x_norm2)
        ffn = tf.keras.layers.Dropout(dropout_rate, name=f'ffn_drop_{i}')(ffn)
        ffn = tf.keras.layers.Dense(d_model, name=f'ffn2_{i}')(ffn)
        x   = tf.keras.layers.Add(name=f'ffn_add_{i}')([x, ffn])

    x      = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='final_norm')(x)
    x_last = x[:, -1, :]    # último token

    # ── Cabezas de salida ────────────────────────────────────────────────────
    out_pitch    = tf.keras.layers.Dense(vocab_size,           name='pitch')(x_last)
    out_step     = tf.keras.layers.Dense(1,                    name='step')(x_last)
    out_duration = tf.keras.layers.Dense(1,                    name='duration')(x_last)
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
# GENERACIÓN DE NOTAS
# ─────────────────────────────────────────────────────────────────────────────
def predict_next_note_ext(notes: np.ndarray,
                          keras_model: tf.keras.Model,
                          temperature: float = 1.0) -> tuple:
    """Muestrea la siguiente nota (pitch, step, duration, velocity)."""
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
# MÉTRICAS ESTADÍSTICAS EXTENDIDAS
# ─────────────────────────────────────────────────────────────────────────────
def compute_stats_extended(original_df: pd.DataFrame,
                            generated_df: pd.DataFrame,
                            results_file) -> None:
    """
    Pruebas estadísticas entre música original y generada.
    Aplica sobre las 3 características primarias (pitch, step, duration)
    más velocity cuando está disponible en ambos DataFrames.
    """
    EPS  = 1e-10
    bins = np.arange(129)

    # ── Pitch ─────────────────────────────────────────────────────────────────
    orig_ph, _ = np.histogram(original_df['pitch'],  bins=bins, density=True)
    gen_ph,  _ = np.histogram(generated_df['pitch'], bins=bins, density=True)
    orig_ph = (orig_ph + EPS) / (orig_ph + EPS).sum()
    gen_ph  = (gen_ph  + EPS) / (gen_ph  + EPS).sum()

    kl_div  = float(entropy(orig_ph, gen_ph))
    js_div  = float(jensenshannon(orig_ph, gen_ph))
    w_pitch = float(wasserstein_distance(original_df['pitch'], generated_df['pitch']))
    ks_p, pv_p = ks_2samp(original_df['pitch'], generated_df['pitch'])

    # Pitch class (mod 12)
    orig_pc = np.bincount(original_df['pitch'].astype(int) % 12, minlength=12).astype(float)
    gen_pc  = np.bincount(generated_df['pitch'].astype(int) % 12, minlength=12).astype(float)
    orig_pc /= orig_pc.sum() + EPS
    gen_pc  /= gen_pc.sum()  + EPS
    js_pc    = float(jensenshannon(orig_pc, gen_pc))

    # ── Step ──────────────────────────────────────────────────────────────────
    ks_s, pv_s   = ks_2samp(original_df['step'], generated_df['step'])
    mw_s, mwp_s  = mannwhitneyu(original_df['step'], generated_df['step'], alternative='two-sided')
    w_step       = float(wasserstein_distance(original_df['step'], generated_df['step']))
    min_len_s    = min(len(original_df), len(generated_df))
    pr_s, _      = pearsonr(
        original_df['step'].values[:min_len_s],
        generated_df['step'].values[:min_len_s]
    ) if min_len_s >= 2 else (float('nan'), None)

    # ── Duration ──────────────────────────────────────────────────────────────
    ks_d, pv_d   = ks_2samp(original_df['duration'], generated_df['duration'])
    mw_d, mwp_d  = mannwhitneyu(original_df['duration'], generated_df['duration'], alternative='two-sided')
    w_dur        = float(wasserstein_distance(original_df['duration'], generated_df['duration']))
    pr_d, _      = pearsonr(
        original_df['duration'].values[:min_len_s],
        generated_df['duration'].values[:min_len_s]
    ) if min_len_s >= 2 else (float('nan'), None)

    # ── Velocity ────────────────────────────────────────
    vel_stats = ""
    if 'velocity' in original_df.columns and 'velocity' in generated_df.columns:
        ks_v, pv_v   = ks_2samp(original_df['velocity'], generated_df['velocity'])
        w_vel        = float(wasserstein_distance(original_df['velocity'], generated_df['velocity']))
        vel_stats    = (
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

    # Estadísticos descriptivos
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
    feats = ['pitch', 'step', 'duration', 'velocity']
    fig, axes = plt.subplots(1, len(feats), figsize=(20, 4))
    colors_o = ['steelblue', 'seagreen', 'coral',      'mediumpurple']
    colors_g = ['orange',    'tomato',   'dodgerblue',  'gold']

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


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == '__main__':

    # ── 1. Descubrimiento de archivos ─────────────────────────────────────────
    filenames = glob.glob(os.path.join(ROOT_DIR, '**/*.mid*'), recursive=True)
    print(f"\nArchivos MIDI encontrados: {len(filenames)}")
    if not filenames:
        raise FileNotFoundError(f"No se encontraron archivos en {ROOT_DIR}")

    # Gráficas de muestra
    try:
        raw_sample = midi_to_notes_extended(filenames[0])
        if not raw_sample.empty:
            plot_piano_roll(raw_sample)
            plot_feature_distributions(raw_sample, tag='_muestra')
            print("Gráficas de muestra guardadas.")
    except Exception as e:
        print(f"[AVISO] No se pudo graficar la muestra: {e}")

    # ── 2. Extracción masiva de características extendidas ────────────────────
    all_notes = []
    exitosos  = 0
    fallidos  = 0
    total     = len(filenames)

    print(f"\nExtrayendo 8 características de {total} archivos MIDI...")
    for f in filenames:
        try:
            df = midi_to_notes_extended(f)
            if df is not None and len(df) > SEQ_LENGTH:
                all_notes.append(df)
                exitosos += 1
            if exitosos % 100 == 0 and exitosos > 0:
                print(f"  ✓ {exitosos}/{total} archivos procesados")
        except Exception as e:
            fallidos += 1

    print(f"\n  Exitosos : {exitosos}")
    print(f"  Fallidos : {fallidos}")

    all_notes_df = pd.concat(all_notes, ignore_index=True)
    n_notes      = len(all_notes_df)
    print(f" Total de eventos : {n_notes:,}")

    # ── 3. Pipeline de dataset ────────────────────────────────────────────────
    # Construir tensor de características: 8 columnas
    # pitch aquí es INT crudo (0-127); la normalización se hace dentro de split_labels
    feat_arr = np.stack(
        [all_notes_df[col].values for col in FEAT_COLS], axis=1
    ).astype(np.float32)

    notes_ds = tf.data.Dataset.from_tensor_slices(feat_arr)
    seq_ds   = create_sequences_extended(notes_ds, SEQ_LENGTH)

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

    # ── 4. Modelo con soporte multi-GPU ───────────────────────────────────────
    strategy = tf.distribute.MirroredStrategy()
    print(f"Réplicas GPU disponibles: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        model = build_transformer_extended()

        lr_schedule = TransformerLRSchedule(D_MODEL, WARMUP_STEPS)
        optimizer   = tf.keras.optimizers.Adam(
            lr_schedule,
            beta_1   = 0.9,
            beta_2   = 0.98,
            epsilon  = 1e-9,
            clipnorm = 1.0
        )
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
        model.compile(
            optimizer    = optimizer,
            loss         = loss_fns,
            loss_weights = loss_weights
        )

    model.summary()

    # ── 5. Callbacks ──────────────────────────────────────────────────────────
    best_ckpt = os.path.join(RESULTS_DIR, 'mejor_modelo.keras')
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor              = 'val_loss',
            patience             = 15,
            restore_best_weights = True,
            verbose              = 1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath       = best_ckpt,
            monitor        = 'val_loss',
            save_best_only = True,
            verbose        = 1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(RESULTS_DIR, 'log_entrenamiento.csv')
        ),
    ]

    # ── 6. Entrenamiento ──────────────────────────────────────────────────────
    print(f"\nIniciando entrenamiento EXTENDIDO: {EPOCHS} épocas, "
          f"batch={BATCH_SIZE}, seq={SEQ_LENGTH}, features={N_FEATURES}, "
          f"d_model={D_MODEL}, layers={NUM_LAYERS}")

    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = EPOCHS,
        callbacks       = callbacks
    )

    model.save(os.path.join(RESULTS_DIR, 'modelo_final.keras'))
    print("Modelo guardado.")

    # Curvas de pérdida
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.epoch, history.history['loss'],             label='Train Total')
    ax.plot(history.epoch, history.history['val_loss'],         label='Val Total')
    ax.plot(history.epoch, history.history['pitch_loss'],       label='Pitch',    linestyle='--')
    ax.plot(history.epoch, history.history['step_loss'],        label='Step',     linestyle='--')
    ax.plot(history.epoch, history.history['duration_loss'],    label='Duration', linestyle='--')
    ax.plot(history.epoch, history.history['velocity_loss'],    label='Velocity', linestyle=':')
    ax.set_xlabel('Época')
    ax.set_ylabel('Pérdida')
    ax.set_title('Historial de Entrenamiento — V3 EXTENDED')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'historial_entrenamiento.png'), dpi=120)
    plt.close()

    # ── 7. Evaluación en validación ───────────────────────────────────────────
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
        perplexity = math.exp(min(history.history['pitch_loss'][-1], 700))
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

    mse_d = mean_squared_error( y_true_dur, y_pred_dur)
    mae_d = mean_absolute_error(y_true_dur, y_pred_dur)
    r2_d  = r2_score(           y_true_dur, y_pred_dur)

    mse_v = mean_squared_error( y_true_vel, y_pred_vel)
    mae_v = mean_absolute_error(y_true_vel, y_pred_vel)
    r2_v  = r2_score(           y_true_vel, y_pred_vel)

    # ── 8. Generación de música ────────────────────────────────────────────────
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

        # Construir ventana de entrada con las 8 características
        feat_mat = np.stack(
            [raw_ext[col].values for col in FEAT_COLS], axis=1
        ).astype(np.float32)

        idx = np.random.randint(0, len(feat_mat) - SEQ_LENGTH)
        win = feat_mat[idx:idx+SEQ_LENGTH].copy()

        # Normalizar pitch en la ventana de entrada
        win[:, IDX_PITCH] /= float(VOCAB_SIZE)

        generated  = []
        prev_start = 0.0

        for _ in range(NUM_PRED):
            pitch, step, duration, velocity = predict_next_note_ext(win, model, TEMPERATURE)
            start = prev_start + step
            end   = start + duration
            generated.append((pitch, step, duration, velocity, start, end))

            # Nueva fila normalizada (todas las continuas ya en [0,1])
            new_row = np.array([
                pitch / VOCAB_SIZE,   # pitch norm
                step,
                duration,
                velocity,
                0.0,                  # sustain: sin información futura
                min(1.0 / MAX_CHORD, 1.0),  # chord_size: asumimos monofónico
                win[-1, IDX_TEMPO],   # reutilizar el último tempo conocido
                0.0                   # beat_position: reseteado
            ], dtype=np.float32)

            win = np.concatenate([win[1:], new_row[np.newaxis, :]], axis=0)
            prev_start = start

        gen_df = pd.DataFrame(
            generated,
            columns=['pitch', 'step', 'duration', 'velocity', 'start', 'end']
        )
        out_path = os.path.join(OUTPUT_DIR, f'cancion_EXT_{pistas_ok+1:02d}.mid')
        notes_to_midi_extended(gen_df, out_path)

        if pistas_ok == 0:
            first_generated_df = gen_df
            first_original_df  = raw_ext

        print(f"  → Pista {pistas_ok+1}/{NUM_SONGS} guardada en {out_path}")
        pistas_ok += 1

    # ── 9. Guardar métricas ────────────────────────────────────────────────────
    metrics_path = os.path.join(RESULTS_DIR, 'metricas_evaluacion_EXTENDED.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  MÉTRICAS — entrenamiento_v3_EXTENDED\n")
        f.write("=" * 60 + "\n\n")

        f.write("── CLASIFICACIÓN (PITCH) ──\n")
        f.write(f"  Accuracy   : {acc:.4f}\n")
        f.write(f"  F1-Score   : {f1:.4f}\n")
        f.write(f"  Recall     : {rec:.4f}\n")
        f.write(f"  Perplexity : {perplexity:.4f}\n")
        f.write(f"  BLEU-1     : {np.mean(bleu_1):.4f}\n")
        f.write(f"  BLEU-2     : {np.mean(bleu_2):.4f}\n")
        f.write(f"  BLEU-4     : {np.mean(bleu_4):.4f}\n\n")

        f.write("── REGRESIÓN (STEP) ──\n")
        f.write(f"  MSE : {mse_s:.6f} | MAE : {mae_s:.6f} | R² : {r2_s:.4f}\n\n")

        f.write("── REGRESIÓN (DURATION) ──\n")
        f.write(f"  MSE : {mse_d:.6f} | MAE : {mae_d:.6f} | R² : {r2_d:.4f}\n\n")

        f.write("── REGRESIÓN (VELOCITY) ──\n")
        f.write(f"  MSE : {mse_v:.6f} | MAE : {mae_v:.6f} | R² : {r2_v:.4f}\n\n")

        f.write("── PARÁMETROS DEL MODELO ──\n")
        f.write(f"  Características de entrada : {N_FEATURES}\n")
        f.write(f"  d_model={D_MODEL}, heads={NUM_HEADS}, ff_dim={FF_DIM}, "
                f"layers={NUM_LAYERS}, seq_len={SEQ_LENGTH}, dropout={DROPOUT_RATE}\n\n")

        if first_generated_df is not None and first_original_df is not None:
            compute_stats_extended(first_original_df, first_generated_df, f)
            plot_comparison_ext(first_original_df, first_generated_df, tag='_EXTENDED')

    print(f"\n Métricas guardadas en {metrics_path}")