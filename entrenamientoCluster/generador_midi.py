import os
import numpy as np
import pandas as pd
import pretty_midi

# =============================================================================
# MOTOR BASE DE INFERENCIA
# Este script está aislado para no tocar el entrenamiento.
# =============================================================================

def predict_next_note_base(notes: np.ndarray,
                           keras_model,
                           temperature: float = 1.0) -> tuple:
    """Predice la siguiente nota (Versión base sin Top-K)."""
    import tensorflow as tf
    inputs = tf.expand_dims(notes, 0)
    preds = keras_model(inputs, training=False)

    pitch_logits = preds['pitch'][0].numpy() / temperature
    pitch_logits -= np.max(pitch_logits)
    probs = np.exp(pitch_logits)
    probs /= probs.sum()
    
    # 128 es el VOCAB_SIZE por defecto
    pitch = int(np.random.choice(128, p=probs))
    
    step     = max(float(np.maximum(0.0, preds['step'][0, 0].numpy())), 0.01)
    duration = max(float(np.maximum(0.0, preds['duration'][0, 0].numpy())), 0.05)
    velocity = float(np.clip(preds['velocity'][0, 0].numpy(), 0.0, 1.0))
    
    return pitch, step, duration, velocity

def save_to_midi(notes_df: pd.DataFrame, out_file: str):
    """Guarda el DataFrame a un archivo MIDI reproducible."""
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0) # Acoustic Grand Piano
    prev_start = 0.0
    for _, row in notes_df.iterrows():
        start = float(prev_start + row['step'])
        end = float(start + row['duration'])
        vel = int(np.clip(round(row.get('velocity', 0.63) * 127), 1, 127))
        instrument.notes.append(pretty_midi.Note(
            velocity=vel, pitch=int(row['pitch']), start=start, end=end
        ))
        prev_start = start
    pm.instruments.append(instrument)
    pm.write(out_file)
    print(f"MIDI guardado en {out_file}")

if __name__ == '__main__':
    print("Script de Inferencia Base cargado.")
    print("Para generar música, asegúrate de tener los pesos en RESULTADOS_V3_EXTENDED/")
