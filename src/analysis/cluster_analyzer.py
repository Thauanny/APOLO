# Copyright (c) 2025 Thauanny Kyssy Ramos Pereira. Todos os Direitos Reservados.
#
# Este software é propriedade confidencial e proprietária de Thauanny Kyssy Ramos Pereira.
# A utilização, cópia ou divulgação deste ficheiro só é permitida de acordo
# com os termos de um contrato de licença celebrado com o autor.

"""
Este módulo contém a classe ClusterAnalyzer, a única responsável por
aplicar o DBSCAN para análise, treino e deteção de anomalias.
"""
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import joblib

class ClusterAnalyzer:
    """
    Encapsula toda a lógica de clusterização e deteção de anomalias com DBSCAN.
    """

    def __init__(self, eps: float = 1.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self._scaler = StandardScaler()
        self._dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self._feature_columns = None
        self._trained_data = None
        self._normal_cluster_label = None

    @staticmethod
    def _scale_features(features_df: pd.DataFrame):
        """Aplica o StandardScaler às features."""
        return StandardScaler().fit_transform(features_df)

    @staticmethod
    def calculate_k_distance_graph(features_df: pd.DataFrame, k: int):
        """
        Calcula as distâncias para o k-ésimo vizinho mais próximo para
        ajudar a estimar o melhor valor de 'eps'.
        """
        if features_df.empty:
            return np.array([])
            
        scaled_data = ClusterAnalyzer._scale_features(features_df)
        
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(scaled_data)
        distances, indices = neighbors_fit.kneighbors(scaled_data)
        
        sorted_distances = np.sort(distances[:, k-1], axis=0)
        return sorted_distances

    def analyze(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executa uma análise de clusterização num DataFrame e retorna os resultados.
        """
        if features_df.empty:
            return {}

        scaled_features = self._scale_features(features_df)
        clusters = self._dbscan.fit_predict(scaled_features)
        
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)

        principal_components = None
        if scaled_features.shape[1] >= 2:
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(scaled_features)

        return {
            "clusters": clusters,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "principal_components": principal_components
        }

    def fit(self, baseline_df: pd.DataFrame):
        """
        Treina o ClusterAnalyzer com dados de base para aprender o que é 'normal'.
        """
        features_df = baseline_df.drop(columns=['label'], errors='ignore')
        self._feature_columns = features_df.columns.tolist()
        
        scaled_data = self._scaler.fit_transform(features_df)
        labels = self._dbscan.fit_predict(scaled_data)
        
        if len(labels) > 0:
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(counts) > 0:
                self._normal_cluster_label = unique_labels[np.argmax(counts)]
                self._trained_data = scaled_data[labels == self._normal_cluster_label]
                print(f"Linha de base treinada. O cluster de 'normalidade' é o {self._normal_cluster_label}.")
            else:
                self._trained_data = np.array([])
                print("Aviso: Nenhum cluster de normalidade encontrado.")
        else:
            print("Aviso: Nenhum dado para treinar.")

    def predict_is_anomalous(self, features: dict) -> bool:
        """
        Prevê se um novo conjunto de features é uma anomalia.
        """
        if self._trained_data is None: raise RuntimeError("O modelo deve ser treinado com 'fit()' antes de prever.")
        if self._trained_data.shape[0] == 0: return True

        features_df = pd.DataFrame([features])[self._feature_columns]
        scaled_point = self._scaler.transform(features_df)
        
        distances = np.linalg.norm(self._trained_data - scaled_point, axis=1)
        return np.min(distances) > self.eps

    def predict_clusters(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Aplica o conhecimento do modelo treinado a um novo dataset para
        classificar cada ponto como 'normal' ou 'anomalia', de forma rápida.
        """
        if self._trained_data is None: raise RuntimeError("O modelo deve ser treinado com 'fit()' antes de prever.")
        scaled_data = self._scaler.transform(features_df)
        labels = np.full(shape=len(scaled_data), fill_value=-1, dtype=int)
        if self._trained_data.shape[0] > 0:
            for i, point in enumerate(scaled_data):
                distances = np.linalg.norm(self._trained_data - point, axis=1)
                if np.min(distances) <= self.eps:
                    labels[i] = self._normal_cluster_label
        return labels

    def save_model(self, path: str):
        """Salva o estado do analyzer treinado num ficheiro."""
        joblib.dump(self, path)
        print(f"Analyzer salvo em {path}")

    @staticmethod
    def load_model(path: str):
        """Carrega um analyzer treinado a partir de um ficheiro."""
        analyzer = joblib.load(path)
        print(f"Analyzer carregado de {path}")
        return analyzer