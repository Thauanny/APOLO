import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

def plot_test_results(
    time_axis: List[float],
    sensor_data: List[float],
    fft_results: Tuple[np.ndarray, np.ndarray, float, float],
    test_name: str,
    sensor_axis: str = "Aceleração Eixo X (g)"
) -> plt.Figure:
    """
    Cria e retorna uma figura Matplotlib com os resultados do teste.
    """
    
    fft_x, fft_y, dominant_freq, max_amplitude = fft_results

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f"Resultados do Teste: {test_name}", fontsize=16)
    
    ax1.plot(time_axis, sensor_data, label=f"Dados do Sensor ({sensor_axis})")
    ax1.set_title("Sinal do Sensor no Tempo")
    ax1.set_xlabel("Tempo (s)")
    ax1.set_ylabel("Amplitude do Sensor (g)")
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(fft_x, fft_y, label="Espectro de Frequência")
    ax2.axvspan(4, 8, color='red', alpha=0.2, label="Faixa de Tremor (4-8 Hz)")
    if dominant_freq > 0:
        ax2.plot(dominant_freq, max_amplitude, 'ro', markersize=10)
        ax2.annotate(f"Pico: {dominant_freq:.2f} Hz", (dominant_freq, max_amplitude),
                     textcoords="offset points", xytext=(0,10), ha='center', color='red')
    ax2.set_title("Análise de Frequência (FFT)")
    ax2.set_xlabel("Frequência (Hz)")
    ax2.set_ylabel("Amplitude")
    ax2.set_xlim(0, 20)
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig