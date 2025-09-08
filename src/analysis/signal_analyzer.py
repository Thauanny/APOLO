import numpy as np
from scipy.fft import fft, fftfreq
from typing import List, Tuple

# Frequências típicas de tremor de repouso (Parkinson) em Hz
TREMOR_FREQ_MIN = 4.0
TREMOR_FREQ_MAX = 8.0

class SignalAnalyzer:
    """
    Analisa uma série temporal de dados de sensores para detectar
    frequências de tremor.
    """

    @staticmethod
    def find_tremor_frequency(
        sensor_readings: List[float], 
        sample_rate: float
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Aplica a Transformada Rápida de Fourier (FFT) para encontrar a frequência
        dominante em um sinal.

        Args:
            sensor_readings: Lista de leituras de um eixo do sensor (ex: aceleração em X).
            sample_rate: A taxa de amostragem em Hz (leituras por segundo).

        Returns:
            Uma tupla contendo:
            - Frequências (eixo x do gráfico de FFT)
            - Amplitude da FFT (eixo y do gráfico de FFT)
            - Frequência dominante na faixa de tremor
            - Amplitude dessa frequência dominante
        """
        n = len(sensor_readings)
        if n == 0 or sample_rate <= 0:
            return np.array([]), np.array([]), 0.0, 0.0

        # Normaliza o sinal (remove a média)
        normalized_signal = np.array(sensor_readings) - np.mean(sensor_readings)

        # Calcula a FFT
        yf = fft(normalized_signal)
        xf = fftfreq(n, 1 / sample_rate)

        # Pega apenas as frequências positivas
        positive_mask = xf > 0
        xf = xf[positive_mask]
        # Pega a magnitude (amplitude) e normaliza
        yf = 2.0/n * np.abs(yf[positive_mask])

        # Filtra para encontrar a frequência de pico na faixa de tremor
        tremor_mask = (xf >= TREMOR_FREQ_MIN) & (xf <= TREMOR_FREQ_MAX)
        
        dominant_freq = 0.0
        max_amplitude = 0.0

        if np.any(tremor_mask):
            freqs_in_range = xf[tremor_mask]
            amps_in_range = yf[tremor_mask]
            
            if len(amps_in_range) > 0:
                max_amp_index = np.argmax(amps_in_range)
                dominant_freq = freqs_in_range[max_amp_index]
                max_amplitude = amps_in_range[max_amp_index]

        return xf, yf, dominant_freq, max_amplitude