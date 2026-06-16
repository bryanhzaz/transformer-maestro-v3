import os
import numpy as np
import pretty_midi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_piano_roll_2d(midi_path: str, output_path: str, max_time: float = 30.0):
    """
    Toma un archivo MIDI y genera un "Piano Roll" (visualización 2D de las notas).
    El Eje Y es el Pitch (Nota), el Eje X es el tiempo.
    Excelente para entender cómo el Transformer ve los datos crudos.
    """
    if not os.path.exists(midi_path):
        print(f"[!] No existe {midi_path}")
        return

    pm = pretty_midi.PrettyMIDI(midi_path)
    # Extraer el piano roll: shape = (128, frames)
    # sampling frequency = 100 Hz (100 frames por segundo)
    fs = 100
    piano_roll = pm.get_piano_roll(fs=fs)
    
    # Recortar al max_time
    max_frames = int(max_time * fs)
    piano_roll = piano_roll[:, :max_frames]
    
    # Preparar el plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Binarizar para visualización (solo importa si hay nota o no, no la velocidad)
    piano_roll_bin = np.where(piano_roll > 0, 1, 0)
    
    ax.imshow(piano_roll_bin, aspect='auto', origin='lower', cmap='Blues', 
              extent=[0, piano_roll.shape[1]/fs, 0, 127])
              
    ax.set_title(f"Piano Roll - {os.path.basename(midi_path)} (Primeros {max_time}s)")
    ax.set_xlabel("Tiempo (segundos)")
    ax.set_ylabel("Pitch (Nota MIDI)")
    ax.set_ylim(20, 100) # Rango típico de un piano real
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Piano Roll guardado en {output_path}")

if __name__ == '__main__':
    # Archivo de prueba
    midi_prueba = 'composicion_ia_1.mid'
    if not os.path.exists(midi_prueba):
        print("Crea o proporciona un archivo MIDI para visualizar.")
    else:
        plot_piano_roll_2d(midi_prueba, 'piano_roll_ejemplo.png')
