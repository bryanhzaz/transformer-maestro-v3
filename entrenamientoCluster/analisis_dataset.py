import os
import glob
import numpy as np
import pretty_midi

def escanear_dataset(ruta_carpeta: str, max_archivos: int = 50):
    """
    Escanea una carpeta de archivos MIDI y extrae estadísticas rápidas.
    Ideal para el Análisis Exploratorio de Datos (EDA).
    """
    archivos = glob.glob(os.path.join(ruta_carpeta, '**/*.mid*'), recursive=True)
    if not archivos:
        print(f"No se encontraron archivos en {ruta_carpeta}")
        return

    print(f"Analizando los primeros {min(len(archivos), max_archivos)} archivos de {len(archivos)} totales...\n")
    
    duraciones = []
    total_notas = []
    
    for f in archivos[:max_archivos]:
        try:
            pm = pretty_midi.PrettyMIDI(f)
            duracion = pm.get_end_time()
            notas = sum(len(inst.notes) for inst in pm.instruments)
            
            duraciones.append(duracion)
            total_notas.append(notas)
        except Exception as e:
            print(f"Error al leer {f}: {e}")

    if duraciones:
        print("=== RESULTADOS DEL EDA ===")
        print(f"Duración promedio por pista: {np.mean(duraciones):.2f} segundos")
        print(f"Notas promedio por pista:    {np.mean(total_notas):.0f} notas")
        print(f"Densidad promedio:           {np.mean(total_notas)/np.mean(duraciones):.2f} notas/segundo")
        print("==========================")

if __name__ == '__main__':
    # Asume que la carpeta de datos maestro está un nivel arriba
    carpeta_maestro = '../maestro-v3.0.0'
    escanear_dataset(carpeta_maestro)
