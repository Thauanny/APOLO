# Copyright (c) 2025 Thauanny Kyssy Ramos Pereira. Todos os Direitos Reservados.
# ... (cabeçalho de copyright igual)

"""
Este módulo contém a classe para deteção de anomalias usando o algoritmo DBSCAN.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib

class AnomalyDetector:
    """
    Encapsula a lógica para treinar um modelo DBSCAN numa linha de base
    e prever se novos dados são anómalos.
    """

    def __init__(self, eps: float = 1.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self._scaler = StandardScaler()
        self._dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self._feature_columns = None
        self._trained_data = None
        self._normal_cluster_label = None

    def fit(self, baseline_df: pd.DataFrame):
        """
        Treina o detetor com dados de base para aprender o que é 'normal'.
        """
        features_df = baseline_df.drop(columns=['label'], errors='ignore')
        self._feature_columns = features_df.columns.tolist()
        
        scaled_data = self._scaler.fit_transform(features_df)
        labels = self._dbscan.fit_predict(scaled_data)
        
        # Assume que o maior cluster encontrado é o da "normalidade"
        if len(labels) > 0:
            counts = np.bincount(labels[labels != -1])
            if len(counts) > 0:
                self._normal_cluster_label = np.argmax(counts)
                # Guarda apenas os pontos que pertencem ao cluster de normalidade
                self._trained_data = scaled_data[labels == self._normal_cluster_label]
                print(f"Linha de base treinada. O cluster de 'normalidade' é o {self._normal_cluster_label}.")
            else: # Caso onde só encontrou ruído
                self._trained_data = np.array([])
                print("Aviso: Nenhum cluster de normalidade encontrado nos dados de base.")
        else:
            print("Aviso: Nenhum dado fornecido para treinar a linha de base.")

    def predict_is_anomalous(self, features: dict) -> bool:
        """
        Prevê se um novo conjunto de features é uma anomalia.
        Retorna True se for uma anomalia, False se for normal.
        """
        if self._trained_data is None:
            raise RuntimeError("O modelo deve ser treinado com 'fit()' antes de prever.")
        
        if self._trained_data.shape[0] == 0:
            return True # Se não há 'normalidade', tudo é anómalo

        features_df = pd.DataFrame([features])[self._feature_columns]
        scaled_point = self._scaler.transform(features_df)
        
        # Lógica de previsão: verifica se o novo ponto está perto (dentro de 'eps')
        # de algum ponto do cluster de normalidade aprendido.
        distances = np.linalg.norm(self._trained_data - scaled_point, axis=1)
        
        # Se a distância mínima for maior que 'eps', está fora do cluster, é uma anomalia.
        return np.min(distances) > self.eps

    def save_model(self, path: str):
        joblib.dump(self, path)
        print(f"Modelo salvo em {path}")

    @staticmethod
    def load_model(path: str):
        detector = joblib.load(path)
        print(f"Modelo carregado de {path}")
        return detector