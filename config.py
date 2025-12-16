# Copyright (c) 2025 Thauanny Kyssy Ramos Pereira. Todos os Direitos Reservados.
#
# Este software é propriedade confidencial e proprietária de Thauanny Kyssy Ramos Pereira.
# A utilização, cópia ou divulgação deste ficheiro só é permitida de acordo
# com os termos de um contrato de licença celebrado com o autor.

"""
Arquivo de constantes e configurações globais do sistema APOLO.
Centraliza todos os parâmetros ajustáveis para fácil manutenção.
"""

# ============================================================================
# DBSCAN - PARÂMETROS DE CLUSTERIZAÇÃO
# ============================================================================

# Raio de vizinhança em espaço 7-dimensional normalizado
DBSCAN_EPS = 1.5

# Mínimo de pontos para considerar um core-point
# Calculado com a heurística: 2 × número_de_dimensões
# Para 7 features: 2 × 7 = 14
DBSCAN_MIN_SAMPLES = 2  # Multiplicador (será multiplicado por num_features)

# ============================================================================
# ANÁLISE DE TREMOR - FAIXAS DE FREQUÊNCIA
# ============================================================================

# Frequência mínima característica de tremor de repouso (Parkinson) em Hz
TREMOR_FREQ_MIN = 4.0

# Frequência máxima característica de tremor de repouso (Parkinson) em Hz
TREMOR_FREQ_MAX = 8.0

# ============================================================================
# PROCESSAMENTO DE SINAIS
# ============================================================================

# Taxa de amostragem alvo em Hz (controlada por time.sleep(0.01) = 10ms)
TARGET_SAMPLE_RATE = 100

# Intervalo de polling em segundos (time.sleep)
POLLING_INTERVAL_SEC = 0.01

# ============================================================================
# SEGMENTAÇÃO EM JANELAS
# ============================================================================

# Tamanho da janela de análise em segundos
WINDOW_SIZE_SEC = 1.0

# Taxa de overlap das janelas (0.5 = 50%)
WINDOW_OVERLAP = 0.5

# ============================================================================
# TESTES E COLETA
# ============================================================================

# Duração de cada teste em segundos
TEST_DURATION_SEC = 10

# Duração recomendada para treino (múltiplas sessões de TEST_DURATION_SEC)
RECOMMENDED_TRAINING_DURATION_MIN = 15

# ============================================================================
# CAMINHOS DE ARQUIVOS
# ============================================================================

# Caminho do dataset de treino
DATASET_PATH = "gameplay_session.csv"

# Caminho do modelo treinado
MODEL_PATH = "analyzer_model.joblib"

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def get_min_samples_for_dimensions(num_features: int) -> int:
    """
    Calcula o min_samples para DBSCAN baseado no número de features.
    
    Heurística: min_samples = DBSCAN_MIN_SAMPLES × num_features
    
    Args:
        num_features: Número de dimensões/features
        
    Returns:
        int: Valor calculado de min_samples
        
    Exemplo:
        >>> get_min_samples_for_dimensions(7)
        14
    """
    return DBSCAN_MIN_SAMPLES * num_features
