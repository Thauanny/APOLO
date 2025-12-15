# Copyright (c) 2025 Thauanny Kyssy Ramos Pereira. Todos os Direitos Reservados.
# ... (cabeçalho)

"""
Este módulo contém a classe SessionProcessor, responsável por transformar
dados brutos de uma sessão de movimento em um DataFrame de features.
"""
import pandas as pd
from src.analysis.signal_analyzer import SignalAnalyzer
from src.analysis.feature_extractor import _extract_features_from_rest_test

class SessionProcessor:
    """
    Processa uma sessão de dados brutos, segmenta-a em janelas
    e extrai features de cada janela.
    """
    def __init__(self, window_size_sec: float = 2.0, sample_rate_hz: int = 100, overlap: float = 0.5):
        self.window_size_sec = window_size_sec
        self.sample_rate_hz = sample_rate_hz
        self.window_size_samples = int(self.window_size_sec * self.sample_rate_hz)
        self.step = int(self.window_size_samples * (1 - overlap))
        if self.step == 0: self.step = 1

    def process_session_df(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Recebe um DataFrame bruto de uma sessão e retorna um DataFrame de features.
        """
        all_features = []
        fft_analyzer = SignalAnalyzer()
        
        accel_columns = ['accel_x', 'Accel_X', 'ACCEL_X', 'acceleration_x', 'ax']
        accel_col = None
        for col in accel_columns:
            if col in raw_df.columns:
                accel_col = col
                break
        
        if accel_col is None:
            print(f"ERRO: Coluna de aceleração não encontrada. Colunas disponíveis: {list(raw_df.columns)}")
            return pd.DataFrame()
        
        signal = raw_df[accel_col].to_numpy()
        
        if len(signal) < self.window_size_samples:
            print("Aviso: A sessão de dados é mais curta que a janela de análise.")
            return pd.DataFrame()

        print(f"Processando {len(signal)} amostras em janelas de {self.window_size_samples} com passo de {self.step}...")
        for i in range(0, len(signal) - self.window_size_samples, self.step):
            window = signal[i:i + self.window_size_samples]
            fft_results = fft_analyzer.find_tremor_frequency(window, self.sample_rate_hz)
            test_result = {
                "name": f"Janela_{i}", "readings": window, 
                "sample_rate": self.sample_rate_hz, "fft_results": fft_results
            }
            features = _extract_features_from_rest_test(test_result)
            all_features.append(features)

        if not all_features:
            return pd.DataFrame()
            
        return pd.DataFrame(all_features)