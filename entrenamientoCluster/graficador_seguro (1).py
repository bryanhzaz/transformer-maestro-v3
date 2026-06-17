import os
import pandas as pd
import matplotlib
# Configurar backend Agg para evitar cuelgues si no hay pantalla (ideal para clusters)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_seguro_curvas(log_path: str, output_path: str):
    """
    Genera la curva de entrenamiento utilizando un manejo explícito
    del objeto figura y limpiando la RAM explícitamente para evitar Memory Leaks.
    """
    if not os.path.exists(log_path):
        print(f"[!] No se encontró {log_path} para graficar.")
        return

    df = pd.read_csv(log_path)
    
    # Manejo seguro: instanciar figura explícitamente
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(df['epoch'], df['loss'], label='Entrenamiento', color='steelblue', lw=2)
    if 'val_loss' in df.columns:
        ax.plot(df['epoch'], df['val_loss'], label='Validación', color='orange', lw=2)
        
    ax.set_title('Convergencia del Modelo (Uso de Memoria Seguro)')
    ax.set_xlabel('Época')
    ax.set_ylabel('Pérdida (Loss)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Guardar y destruir la figura (Evita memory leaks)
    fig.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfica guardada en {output_path}")

if __name__ == '__main__':
    log_file = 'RESULTADOS_V3_EXTENDED/log_entrenamiento.csv'
    out_img = 'RESULTADOS_V3_EXTENDED/curva_segura.png'
    os.makedirs('RESULTADOS_V3_EXTENDED', exist_ok=True)
    plot_seguro_curvas(log_file, out_img)
