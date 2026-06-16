import os
import pandas as pd

def generar_reporte_md(log_path: str, output_path: str):
    """
    Toma el archivo CSV de logs de Keras y genera un reporte Markdown
    ideal para entregar como evidencia de validación.
    """
    if not os.path.exists(log_path):
        print(f"[!] No se encontró {log_path}. Asegúrate de tener los logs.")
        return

    df = pd.read_csv(log_path)
    mejor_epoca = df.loc[df['val_loss'].idxmin()]
    
    contenido = f"""# Reporte de Validación del Transformer Musical

Este documento fue generado automáticamente a partir de los logs de entrenamiento.

## 📊 Métricas Finales
- **Épocas completadas:** {len(df)}
- **Época óptima (Menor Val Loss):** {int(mejor_epoca['epoch'])}
- **Pérdida final (Entrenamiento):** {df['loss'].iloc[-1]:.4f}
- **Pérdida final (Validación):** {df['val_loss'].iloc[-1]:.4f}

## 📝 Detalles por Característica (Época Óptima)
- **Pitch Loss:** {mejor_epoca.get('val_pitch_loss', 'N/A')}
- **Step Loss:** {mejor_epoca.get('val_step_loss', 'N/A')}
- **Duration Loss:** {mejor_epoca.get('val_duration_loss', 'N/A')}
- **Velocity Loss:** {mejor_epoca.get('val_velocity_loss', 'N/A')}

*Reporte generado por exportador_reportes.py*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(contenido)
    print(f"Reporte MD guardado en {output_path}")

if __name__ == '__main__':
    log_file = 'RESULTADOS_V3_EXTENDED/log_entrenamiento.csv'
    out_file = 'RESULTADOS_V3_EXTENDED/reporte_evaluacion.md'
    os.makedirs('RESULTADOS_V3_EXTENDED', exist_ok=True)
    generar_reporte_md(log_file, out_file)
