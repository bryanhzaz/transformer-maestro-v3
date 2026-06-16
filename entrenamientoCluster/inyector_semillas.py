import numpy as np

# =============================================================================
# FUNCIONES AVANZADAS DE MUESTREO (TOP-K) E INYECCIÓN DE SEMILLAS
# =============================================================================

def top_k_sampling(logits: np.ndarray, k: int = 10, temperature: float = 1.0) -> int:
    """
    Realiza un muestreo Top-K para mejorar la armonía musical generada.
    Filtra todas las notas excepto las 'k' más probables.
    """
    logits = logits / temperature
    top_k_indices = np.argsort(logits)[-k:]
    top_k_logits = logits[top_k_indices]
    
    # Softmax sobre el subset
    top_k_logits -= np.max(top_k_logits)
    probs = np.exp(top_k_logits)
    probs /= probs.sum()
    
    chosen_index = np.random.choice(len(probs), p=probs)
    return top_k_indices[chosen_index]

def inyectar_semilla_manual(semilla: np.ndarray, seq_length: int = 256) -> np.ndarray:
    """
    Permite al usuario inyectar un motivo musical (ej. primeras notas de Fur Elise)
    para que el modelo lo continúe. Rellena el resto de la ventana con ceros válidos.
    """
    secuencia_base = np.zeros((seq_length, 8), dtype=np.float32)
    # Rellenar con una velocidad estándar si la secuencia está vacía
    secuencia_base[:, 3] = 0.6 
    
    longitud_semilla = min(len(semilla), seq_length)
    if longitud_semilla > 0:
        secuencia_base[-longitud_semilla:] = semilla[-longitud_semilla:]
        print(f"Semilla inyectada: {longitud_semilla} notas.")
        
    return secuencia_base

if __name__ == '__main__':
    print("Módulo de Inyección de Semillas y Top-K listo para importar.")
    
    # Ejemplo de semilla falsa (4 notas C mayor)
    semilla_ejemplo = np.zeros((4, 8))
    semilla_ejemplo[:, 0] = [60, 64, 67, 72] # C, E, G, C
    secuencia_lista = inyectar_semilla_manual(semilla_ejemplo)
    print("Última nota inyectada en la ventana:", secuencia_lista[-1][0])
