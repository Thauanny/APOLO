# Copyright (c) 2025 Thauanny Kyssy Ramos Pereira. Todos os Direitos Reservados.
#
# Este software é propriedade confidencial e proprietária de Thauanny Kyssy Ramos Pereira.
# A utilização, cópia ou divulgação deste ficheiro só é permitida de acordo
# com os termos de um contrato de licença celebrado com o autor.

"""
Este módulo contém a classe ClusterAnalyzer, a única responsável por
aplicar o DBSCAN para análise, treino e deteção de anomalias.
Implementa padrão Singleton para garantir uma única instância em toda aplicação.
"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import joblib
from config import DBSCAN_EPS, get_min_samples_for_dimensions

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

class ClusterAnalyzer:
    """
    Encapsula toda a lógica de clusterização e deteção de anomalias com DBSCAN.
    Implementa padrão Singleton: apenas uma instância em toda a aplicação.
    """
    
    _instance = None  # Instância singleton
    
    def __new__(cls, *args, **kwargs):
        """Garante que apenas uma instância seja criada."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, eps: float = None, min_samples: int = None):
        """
        Inicializa o ClusterAnalyzer com parâmetros DBSCAN.
        Singleton: Se já foi inicializado, esta chamada é ignorada.
        
        Args:
            eps: Raio de vizinhança (default: DBSCAN_EPS de config.py)
            min_samples: Mínimo de pontos para core-point (default: 2 × num_features)
        """
        # Evita reinicialização se já foi feita
        if self._initialized:
            return
        
        self.eps = eps if eps is not None else DBSCAN_EPS
        # min_samples será calculado no fit() quando soubermos o num_features
        self.min_samples = min_samples
        self._scaler = StandardScaler()
        self._dbscan = None  # Será inicializado no fit()
        self._feature_columns = None
        self._trained_data = None
        self._normal_cluster_label = None
        self._initialized = True
        
        print("[ClusterAnalyzer] Singleton inicializado")

    @staticmethod
    def _scale_features(features_df: pd.DataFrame):
        """Aplica o StandardScaler às features."""
        return StandardScaler().fit_transform(features_df)

    @staticmethod
    def reduce_dimensions_pca(features_df: pd.DataFrame, n_components: int = 2) -> np.ndarray:
        """
        Reduz dimensionalidade usando PCA (rápido, linear).
        """
        if features_df.empty or features_df.shape[1] < 2:
            return None
        
        scaled_data = ClusterAnalyzer._scale_features(features_df)
        n_components = min(n_components, scaled_data.shape[1])
        pca = PCA(n_components=n_components)
        return pca.fit_transform(scaled_data)

    @staticmethod
    def get_pca_variance_explained(features_df: pd.DataFrame) -> tuple:
        """
        Retorna a variância explicada por cada componente do PCA.
        Returns:
            tuple: (componentes, variância_individual, variância_acumulada)
        """
        if features_df.empty or features_df.shape[1] < 2:
            return None, None, None
        
        scaled_data = ClusterAnalyzer._scale_features(features_df)
        pca = PCA()
        pca.fit(scaled_data)
        
        n_components = len(pca.explained_variance_ratio_)
        componentes = list(range(1, n_components + 1))
        variancia_individual = pca.explained_variance_ratio_ * 100
        variancia_acumulada = np.cumsum(pca.explained_variance_ratio_) * 100
        
        return componentes, variancia_individual, variancia_acumulada

    @staticmethod
    def reduce_dimensions_tsne(features_df: pd.DataFrame, n_components: int = 2, perplexity: int = 30) -> np.ndarray:
        """
        Reduz dimensionalidade usando t-SNE (não-linear, interpretável para visualização).
        Melhor para exploração de clusters mas mais lento.
        """
        if features_df.empty or features_df.shape[1] < 2:
            return None
        
        scaled_data = ClusterAnalyzer._scale_features(features_df)
        # Ajustar perplexity para amostras pequenas
        n_samples = scaled_data.shape[0]
        perplexity = min(perplexity, (n_samples - 1) // 3)
        perplexity = max(5, perplexity)
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        return tsne.fit_transform(scaled_data)

    @staticmethod
    def reduce_dimensions_umap(features_df: pd.DataFrame, n_components: int = 2, n_neighbors: int = 15) -> Optional[np.ndarray]:
        """
        Reduz dimensionalidade usando UMAP (não-linear, rápido, preserva estrutura global).
        Requer instalação: pip install umap-learn
        """
        if not HAS_UMAP:
            print("Aviso: UMAP não está instalado. Use: pip install umap-learn")
            return None
        
        if features_df.empty or features_df.shape[1] < 2:
            return None
        
        scaled_data = ClusterAnalyzer._scale_features(features_df)
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42, metric='euclidean')
        return reducer.fit_transform(scaled_data)

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
        distances, _ = neighbors_fit.kneighbors(scaled_data)
        
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
        Considera TODOS os clusters (exceto ruído/-1) como normalidade.
        """
        features_df = baseline_df.drop(columns=['label'], errors='ignore')
        self._feature_columns = features_df.columns.tolist()
        
        # Calcula min_samples dinamicamente se não foi especificado
        if self.min_samples is None:
            num_features = features_df.shape[1]
            self.min_samples = get_min_samples_for_dimensions(num_features)
            print(f"min_samples calculado automaticamente: {self.min_samples} (2 × {num_features} features)")
        
        # Inicializa DBSCAN com os parâmetros
        self._dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        
        scaled_data = self._scaler.fit_transform(features_df)
        labels = self._dbscan.fit_predict(scaled_data)
        
        if len(labels) > 0:
            valid_labels = labels[labels != -1]
            if len(valid_labels) > 0:
                self._trained_data = scaled_data[labels != -1]
                n_clusters = len(np.unique(valid_labels))
                n_normal_points = len(valid_labels)
                print(f"Linha de base treinada. {n_clusters} cluster(s) com {n_normal_points} pontos de 'normalidade'.")
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
        classificar cada ponto como 'normal' (0) ou 'anomalia' (-1).
        """
        if self._trained_data is None: 
            raise RuntimeError("O modelo deve ser treinado com 'fit()' antes de prever.")
        scaled_data = self._scaler.transform(features_df)
        labels = np.full(shape=len(scaled_data), fill_value=-1, dtype=int)
        if self._trained_data.shape[0] > 0:
            for i, point in enumerate(scaled_data):
                distances = np.linalg.norm(self._trained_data - point, axis=1)
                if np.min(distances) <= self.eps:
                    labels[i] = 0
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