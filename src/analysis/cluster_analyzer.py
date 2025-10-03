# Copyright (c) 2025 Thauanny Kyssy Ramos Pereira. Todos os Direitos Reservados.
#
# Este software é propriedade confidencial e proprietária de Thauanny Kyssy Ramos Pereira.
# A utilização, cópia ou divulgação deste ficheiro só é permitida de acordo
# com os termos de um contrato de licença celebrado com o autor.

"""
Este módulo contém a classe ClusterAnalyzer, responsável por aplicar
algoritmos de clusterização como o DBSCAN a um conjunto de dados de features.
"""
from typing import Dict, Any
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

class ClusterAnalyzer:
    """
    Encapsula a lógica de pré-processamento, clusterização e redução
    de dimensionalidade para a análise de anomalias.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=2)
        self._dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)

    def analyze(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executa o pipeline completo de análise no DataFrame de features.

        Args:
            features_df: DataFrame contendo apenas as colunas de features numéricas.

        Returns:
            Um dicionário contendo os resultados da análise.
        """
        if features_df.empty:
            return {}

        # 1. Escala dos dados
        scaled_features = self._scaler.fit_transform(features_df)

        # 2. Aplicação do DBSCAN
        clusters = self._dbscan.fit_predict(scaled_features)

        # 3. Análise de Resultados
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)

        # 4. Redução de Dimensionalidade para Visualização
        if scaled_features.shape[1] < 2:
            principal_components = None
        else:
            principal_components = self._pca.fit_transform(scaled_features)

        # Retorna um dicionário com todos os resultados prontos para a UI
        return {
            "clusters": clusters,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "principal_components": principal_components
        }