# src/analysis/feature_extractor.py

from typing import Dict, List
import numpy as np
from src.analysis.signal_analyzer import SignalAnalyzer

def _extract_features_from_rest_test(test_result: Dict) -> Dict:
    """Extrai features de um teste de tremor de repouso."""
    sensor_readings = np.array(test_result.get('readings', []))
    sample_rate = test_result.get('sample_rate', 0)
    
    if sensor_readings.size == 0 or sample_rate <= 0:
        return {"peak_freq": 0, "tremor_power": 0, "total_power": 0, "tremor_index": 0}

    # Verifica se sinal é válido (tem variação mínima)
    signal_std = np.std(sensor_readings)
    if signal_std < 0.1:  # Sinal muito plano, retorna valores neutros
        print(f"⚠️ AVISO: Sinal muito plano (std={signal_std:.4f}). Pode indicar dados inválidos ou controle desconectado.")
        return {"peak_freq": 0, "tremor_power": 0, "total_power": 0, "tremor_index": 0}

    analyzer = SignalAnalyzer()
    fft_x, yf, dominant_freq, _ = analyzer.find_tremor_frequency(sensor_readings, sample_rate)
    
    tremor_mask = (fft_x >= 4.0) & (fft_x <= 8.0)
    tremor_power = np.sum(yf[tremor_mask])
    total_power = np.sum(yf)
    tremor_index = tremor_power / total_power if total_power > 0 else 0

    return {
        "peak_freq": dominant_freq, "tremor_power": tremor_power,
        "total_power": total_power, "tremor_index": tremor_index
    }

def _extract_features_from_tapping_test(test_result: Dict) -> Dict:
    """Extrai features de um teste de finger tapping."""
    press_timestamps = test_result.get('readings', [])
    duration = test_result.get('duration', 0)
    
    if len(press_timestamps) < 2 or duration <= 0:
        return {"tap_count": 0, "tap_freq": 0, "tap_interval_std": 0}

    intervals = np.diff(press_timestamps)
    
    return {
        "tap_count": len(press_timestamps),
        "tap_freq": len(press_timestamps) / duration,
        "tap_interval_std": np.std(intervals) if len(intervals) > 0 else 0
    }

def extract_features(test_result: Dict) -> Dict:
    """
    Função principal que delega a extração de features com base no nome do teste.
    """
    test_name = test_result.get('name', '')
    features = {"label": test_result.get('label', 'unknown')}

    features.update({
        "peak_freq": 0, "tremor_power": 0, "total_power": 0, "tremor_index": 0,
        "tap_count": 0, "tap_freq": 0, "tap_interval_std": 0
    })

    if "Repouso" in test_name:
        features.update(_extract_features_from_rest_test(test_result))
    elif "Tapping" in test_name:
        features.update(_extract_features_from_tapping_test(test_result))
    # Adicionar aqui 'elif' para outros tipos de teste (ex: Pronação-Supinação)
    
    return features