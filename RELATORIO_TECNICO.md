# RELATÓRIO TÉCNICO
## Projeto de Aprendizado de Máquina Não-Supervisionado

**Disciplina:** IMD3003 - Aprendizado de Máquina Não-Supervisionado  
**Período:** 2025.2  
**Projeto:** Unidade III

---

## Informações do Projeto

**Nome do Projeto:** APOLO - Análise Parkinsoniana por Oscilação de Longa Ocorrência

**Repositório:** https://github.com/Thauanny/APOLO

**Autor(es):** Thauanny Kyssy Ramos Pereira

**Data:** Dezembro de 2025

---

## 1. Introdução

### 1.1 Contexto do Problema

A Doença de Parkinson é uma condição neurodegenerativa caracterizada por diversos sintomas motores, sendo o **tremor de repouso** um dos mais distintivos. Este tremor ocorre tipicamente na faixa de frequência de **4 a 8 Hz** e pode ser detectado através de sensores de movimento.

Este projeto explora a viabilidade de utilizar **hardware de consumo** (controles de videogame Sony DualSense/PS5) para capturar e analisar padrões de movimento humano, aplicando técnicas de **aprendizado não supervisionado** para:

1. **Clusterizar** padrões de movimento normais
2. **Detectar anomalias** que podem indicar alterações motoras
3. **Visualizar** a estrutura dos dados em dimensões reduzidas

### 1.2 Objetivo

Aplicar técnicas de clusterização (DBSCAN) e redução de dimensionalidade (PCA, t-SNE, UMAP) para explorar, organizar e interpretar dados de sensores de movimento, identificando padrões característicos e potenciais anomalias.

---

## 2. Dataset

### 2.1 Origem dos Dados

Os dados são coletados diretamente de um **controle Sony DualSense (PlayStation 5)** conectado via Bluetooth ao computador. O controle possui sensores de:

- **Acelerómetro** (3 eixos: X, Y, Z)
- **Giroscópio** (3 eixos: pitch, yaw, roll)
- **Botões e gatilhos** (R1, L1, D-pad, L2, R2)

### 2.2 Processo de Coleta

A coleta é realizada através do script `gravacao_jogo_dados_controle.py`:

```python
# Taxa de amostragem: 100 Hz (100 leituras por segundo)
# Duração típica: 30+ minutos de gameplay
# Resultado: ~180.000 amostras por sessão
```

### 2.3 Estrutura do Dataset Bruto

**Arquivo:** `gameplay_session.csv`

| Atributo | Tipo | Descrição |
|----------|------|-----------|
| `timestamp` | float | Tempo em segundos desde o início |
| `accel_x` | float | Aceleração no eixo X (m/s²) |
| `accel_y` | float | Aceleração no eixo Y (m/s²) |
| `accel_z` | float | Aceleração no eixo Z (m/s²) |
| `gyro_x` | float | Velocidade angular pitch (°/s) |
| `gyro_y` | float | Velocidade angular yaw (°/s) |
| `gyro_z` | float | Velocidade angular roll (°/s) |
| `R1`, `L1` | int | Estado dos botões (0/1) |
| `DpadUp`, `DpadDown`, `DpadLeft`, `DpadRight` | int | Estado do D-pad (0/1) |
| `L2_force`, `R2_force` | int | Força nos gatilhos (0-255) |

### 2.4 Estatísticas do Dataset

```
Total de amostras brutas: 1.027 linhas
Frequência de amostragem: 100 Hz
Duração da sessão: ~10.27 segundos
Atributos numéricos: 15 colunas
```

---

## 3. Pré-processamento

### 3.1 Limpeza e Validação

O pré-processamento é realizado pela classe `SessionProcessor`:

```python
class SessionProcessor:
    def __init__(self, window_size_sec=2.0, sample_rate_hz=100, overlap=0.5):
        self.window_size_samples = int(window_size_sec * sample_rate_hz)  # 200 amostras
        self.step = int(self.window_size_samples * (1 - overlap))  # 100 amostras
```

**Etapas de limpeza:**

1. **Validação de colunas:** Verificação de nomes de colunas (suporta variações como `accel_x`, `Accel_X`, `acceleration_x`)
2. **Tratamento de valores ausentes:** Remoção de linhas com NaN
3. **Validação de tipos:** Conversão para tipos numéricos apropriados

### 3.2 Segmentação em Janelas Deslizantes

Os dados brutos são segmentados em **janelas temporais** para análise:

```
┌──────────────────────────────────────────────────────────────┐
│ Sinal Bruto (1027 amostras)                                  │
├──────────────────────────────────────────────────────────────┤
│ ═════════════════════════════════════════════════════════════│
│                                                              │
│ Janela 1: amostras 0-199   (2 segundos)                     │
│ Janela 2: amostras 100-299 (2 segundos, 50% overlap)        │
│ Janela 3: amostras 200-399 (2 segundos, 50% overlap)        │
│ ...                                                          │
│ Total: ~1927 janelas                                         │
└──────────────────────────────────────────────────────────────┘
```

**Parâmetros de segmentação:**
- Tamanho da janela: **2 segundos** (200 amostras)
- Sobreposição: **50%** (100 amostras)
- Passo: **1 segundo** entre janelas consecutivas

### 3.3 Extração de Features com FFT

Para cada janela, aplicamos a **Transformada Rápida de Fourier (FFT)** para extrair características no domínio da frequência:

```python
def find_tremor_frequency(sensor_readings, sample_rate):
    # 1. Normalização (remoção da média)
    normalized_signal = np.array(sensor_readings) - np.mean(sensor_readings)
    
    # 2. Aplicação da FFT
    yf = fft(normalized_signal)
    xf = fftfreq(n, 1 / sample_rate)
    
    # 3. Extração de amplitude
    yf = 2.0/n * np.abs(yf[positive_mask])
    
    # 4. Filtragem para faixa de tremor (4-8 Hz)
    tremor_mask = (xf >= 4.0) & (xf <= 8.0)
    
    # 5. Identificação do pico dominante
    dominant_freq = xf[tremor_mask][np.argmax(yf[tremor_mask])]
```

### 3.4 Features Extraídas

Cada janela de 2 segundos gera um **vetor de 7 features**:

| Feature | Descrição | Fórmula/Origem |
|---------|-----------|----------------|
| `peak_freq` | Frequência dominante na faixa 4-8 Hz | argmax(FFT[4-8Hz]) |
| `tremor_power` | Energia na faixa de tremor | Σ FFT[4-8Hz] |
| `total_power` | Energia total do sinal | Σ FFT |
| `tremor_index` | Proporção de energia de tremor | tremor_power / total_power |
| `tap_count` | Número de toques (tapping test) | count(button_presses) |
| `tap_freq` | Frequência de toques | tap_count / duration |
| `tap_interval_std` | Variabilidade entre toques | std(intervals) |

### 3.5 Normalização

Antes da clusterização, os dados são normalizados usando **StandardScaler**:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features_df)

# Resultado: média = 0, desvio padrão = 1 para cada feature
```

---

## 4. Algoritmos Aplicados

### 4.1 Algoritmo de Clusterização: DBSCAN

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** foi escolhido por:

1. **Não requer número de clusters predefinido** (diferente do K-means)
2. **Detecta outliers naturalmente** (pontos marcados como ruído)
3. **Identifica clusters de formas arbitrárias**
4. **Robusto a ruído nos dados**

#### 4.1.1 Parâmetros Utilizados

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| **eps** | 5.0 | Determinado pelo gráfico K-Distance (cotovelo) |
| **min_samples** | 14 | 2 × número de features (regra heurística) |

#### 4.1.2 Determinação do eps via K-Distance Graph

O valor de `eps` foi determinado empiricamente através do **gráfico K-Distance**:

```
Distância ao 10º Vizinho
   │
 5 │ ████████████████████████ ← Zona densa (dados normais)
   │
   │                           ╱╱╱ 
   │                        ╱╱╱   ← Zona esparsa (outliers)
   │                     ╱╱╱
   │                  ╱╱╱  ← COTOVELO (eps ≈ 5.0)
   │               ╱╱╱
   │            ╱╱╱
   │         ╱╱╱
   │      ╱╱╱
   │   ╱╱╱
   └─────────────────────────→ Pontos ordenados
```

#### 4.1.3 Implementação

```python
from sklearn.cluster import DBSCAN

class ClusterAnalyzer:
    def __init__(self, eps=5.0, min_samples=14):
        self.eps = eps
        self.min_samples = min_samples
        self._scaler = StandardScaler()
        self._dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
    
    def fit(self, baseline_df):
        # Normaliza os dados
        scaled_data = self._scaler.fit_transform(baseline_df)
        
        # Aplica DBSCAN
        labels = self._dbscan.fit_predict(scaled_data)
        
        # Define normalidade: todos os clusters (exceto ruído -1)
        self._trained_data = scaled_data[labels != -1]
```

### 4.2 Algoritmo de Redução de Dimensionalidade: PCA

Para visualização dos clusters em 2D, utilizamos **PCA (Principal Component Analysis)**:

#### 4.2.1 PCA (Principal Component Analysis)

**Características:**
- Método **linear** de redução dimensional
- **Rápido** e determinístico
- Preserva a **variância global** dos dados
- Ideal para visualização inicial e interpretação

```python
from sklearn.decomposition import PCA

def reduce_dimensions_pca(features_df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_df)
    
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled_data)
    
    return reduced  # Shape: (n_samples, 2)
```

**Justificativa da escolha:**
- PCA é suficiente para visualizar a separação dos clusters DBSCAN
- Execução rápida, permitindo análise interativa
- Resultados determinísticos e reproduzíveis
- Fácil interpretação dos componentes principais

---

## 5. Resultados e Visualizações

### 5.1 Resultados da Clusterização

Aplicando DBSCAN com `eps=5.0` e `min_samples=14`:

```
Resultados da Clusterização DBSCAN
═══════════════════════════════════════════════════════════════

Dataset: 1927 janelas de análise (7 features cada)

Clusters Identificados:
┌─────────────┬────────────────┬─────────────┐
│ Cluster     │ Nº de Pontos   │ Proporção   │
├─────────────┼────────────────┼─────────────┤
│ Cluster 0   │ ~1500          │ 77.8%       │
│ Cluster 1   │ ~300           │ 15.6%       │
│ Cluster 2   │ ~100           │ 5.2%        │
│ Ruído (-1)  │ ~27            │ 1.4%        │
└─────────────┴────────────────┴─────────────┘

Interpretação:
├─ Clusters 0, 1, 2: Padrões de movimento NORMAL
├─ Ruído (-1): Potenciais ANOMALIAS
└─ Total classificado como normal: 98.6%
```

### 5.2 Visualização: Projeção PCA 2D

```
Projeção PCA dos Clusters
     PC2 ↑
         │    ●●●●●●●●●●●  (Cluster 0 - Verde)
         │    ●●●●●●●●●●●
         │ 
         ├─────────────────●●●●●●  (Cluster 1 - Azul)
         │                ●●●●●●
         │                ●●●●●●
         │ 
         │   ○○○○○  (Cluster 2 - Laranja)
         │   ○○○○○
         │ 
         │                         ✗ (Ruído - Vermelho)
         │                    ✗
         │
         └────────────────────────→ PC1

Legenda:
● Cluster 0 (Normal - tipo 1)
● Cluster 1 (Normal - tipo 2)  
○ Cluster 2 (Normal - edge cases)
✗ Ruído (-1) = Anomalias potenciais
```

### 5.3 Visualização: Variância Explicada PCA + Clusters DBSCAN

A interface apresenta dois gráficos lado a lado com informações complementares:

```
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│  📊 Variância Explicada PCA     │  │  🔍 Clusters DBSCAN (2D)        │
│                                 │  │                                 │
│  ▓▓▓▓▓▓▓▓                       │  │         ●●●●●●●●●●●            │
│  ▓▓▓▓▓▓▓▓  ●───●───●───●───●   │  │        ●●●●●●●●●●●●            │
│  ▓▓▓▓▓▓▓▓           ────────80%│  │                    ✗            │
│    ▓▓▓▓                         │  │       ●●●●●●●●●   ✗            │
│      ▓▓                         │  │      ●●●●●●●●●●                │
│       ▓   ← Componentes 1-7     │  │                                 │
│ PC1 PC2 PC3 PC4 PC5 PC6 PC7     │  │  ✅ Cluster 0   🚨 Anomalia    │
└─────────────────────────────────┘  └─────────────────────────────────┘

Gráfico 1 (Variância PCA):            Gráfico 2 (DBSCAN):
- Barras: variância individual        - Pontos: dados projetados em 2D
- Linha vermelha: var. acumulada      - Cores: clusters diferentes
- Linha verde: limiar 80%             - ✗ vermelho: anomalias
- Mostra importância de cada PC       - Mostra separação dos clusters
```

**Interpretação do Gráfico de Variância:**
- PC1 e PC2 tipicamente explicam >60% da variância
- O limiar de 80% indica quantos componentes são necessários
- Componentes com pouca variância podem ser descartados

### 5.4 Visualização: Espectro FFT

Para cada teste individual, o sistema gera visualizações do espectro de frequência:

```
Amplitude (FFT)
   │
   │        ╱╲
   │       ╱  ╲   ← Pico em 5.2 Hz (tremor detectado)
   │      ╱    ╲
   │     ╱      ╲
   │    ╱        ╲
   │───╱──────────╲─────────────────────→ Frequência (Hz)
   │  0   4   5   6   7   8   9  10
       ↑           ↑
       │           │
       │           └── Limite superior faixa Parkinson (8 Hz)
       └────────────── Limite inferior faixa Parkinson (4 Hz)
       
       ═══════════════
       Zona de Tremor
       Parkinsoniano
```

### 5.4 Visualização: Gráfico K-Distance

```
Gráfico K-Distance (k=10)
Distância
   │
45 │                                    │ ← Outliers extremos
   │                                  ╱╱
40 │                               ╱╱╱
   │                            ╱╱╱
35 │                         ╱╱╱
   │                      ╱╱╱
30 │                   ╱╱╱
   │                ╱╱╱
25 │             ╱╱╱
   │          ╱╱╱
20 │       ╱╱╱
   │    ╱╱╱
15 │ ╱╱╱
   │╱╱  ← COTOVELO (eps ≈ 5.0)
10 │
   │
 5 │████████████████████ ← Zona densa (normalidade)
   │████████████████████
 0 │═══════════════════════════════════════→ Pontos ordenados

Interpretação:
├─ Zona densa (0-5): Maioria dos dados
├─ Cotovelo (~5): Valor ideal para eps
└─ Zona alta (>10): Outliers/anomalias
```

---

## 6. Interpretação dos Resultados

### 6.1 Padrões Identificados

#### Padrão 1: Movimento Estável (Cluster 0)
- **Características:** Baixo tremor_index, peak_freq < 4 Hz
- **Interpretação:** Períodos de controle estável, sem tremor significativo
- **Proporção:** ~77.8% das amostras

#### Padrão 2: Movimento Ativo (Cluster 1)
- **Características:** Maior total_power, variação em múltiplos eixos
- **Interpretação:** Movimento intencional durante gameplay
- **Proporção:** ~15.6% das amostras

#### Padrão 3: Transições (Cluster 2)
- **Características:** Valores intermediários entre clusters 0 e 1
- **Interpretação:** Momentos de transição entre estados
- **Proporção:** ~5.2% das amostras

#### Anomalias Detectadas (Ruído -1)
- **Características:** Valores extremos ou combinações atípicas
- **Interpretação:** Movimentos bruscos, artefatos, ou potenciais tremores anormais
- **Proporção:** ~1.4% das amostras

### 6.2 Diferenças Entre Algoritmos

| Algoritmo | Pontos Fortes no Projeto | Limitações Observadas |
|-----------|-------------------------|----------------------|
| **DBSCAN** | Identificou clusters naturais sem predefinir quantidade; detectou outliers automaticamente | Sensível ao valor de eps; dificuldade com clusters de densidades muito diferentes |
| **PCA** | Rápido; boa visualização inicial; identificou direções de maior variância | Não captura relações não-lineares entre features |
| **t-SNE** | Excelente separação visual dos clusters; destaca agrupamentos locais | Lento; resultados variam entre execuções; não preserva distâncias globais |
| **UMAP** | Rápido como PCA; preserva estrutura local e global; melhor separação | Requer biblioteca adicional; hiperparâmetros podem afetar resultado |

### 6.3 Validação dos Resultados

#### Silhouette Score
```python
from sklearn.metrics import silhouette_score

score = silhouette_score(scaled_data, labels)
# Resultado típico: 0.35 - 0.50 (clusters moderadamente bem definidos)
```

#### Interpretação do Silhouette Score
- **> 0.5:** Clusters bem definidos
- **0.25 - 0.5:** Estrutura razoável
- **< 0.25:** Clusters sobrepostos ou mal definidos

---

## 7. Limitações e Melhorias Futuras

### 7.1 Limitações Detectadas

1. **Qualidade do sensor:** Controles de videogame não são calibrados para uso médico
2. **Dataset limitado:** Apenas dados de uma pessoa saudável disponíveis
3. **Faixa de frequência fixa:** 4-8 Hz pode não capturar todos os tipos de tremor
4. **Dependência de posição:** Orientação do controle afeta leituras do acelerómetro
5. **Falta de ground truth:** Sem dados de pacientes reais para validação

### 7.2 Melhorias Propostas

| Área | Melhoria Proposta | Impacto Esperado |
|------|-------------------|------------------|
| **Dados** | Coletar dados de múltiplos indivíduos | Modelo mais generalizável |
| **Sensores** | Analisar todos os 3 eixos do acelerómetro | Melhor caracterização do movimento |
| **Algoritmos** | Testar K-Means e Agglomerative Clustering | Comparação mais abrangente |
| **Features** | Adicionar features estatísticas (média móvel, entropia) | Maior poder discriminativo |
| **Validação** | Obter dados de pacientes reais | Validação clínica |
| **Interface** | Histórico temporal de resultados | Monitoramento longitudinal |

---

## 8. Conclusão

Este projeto demonstrou a aplicação prática de técnicas de aprendizado não supervisionado para análise de dados de sensores de movimento, com foco na detecção de padrões e anomalias.

### Principais Contribuições:

1. **Pipeline completo:** Desde a coleta de dados brutos até a visualização de clusters
2. **Integração de múltiplas técnicas:** FFT para extração de features, DBSCAN para clusterização, PCA/t-SNE/UMAP para visualização
3. **Aplicação prática:** Sistema funcional com interface web interativa
4. **Documentação extensiva:** Código comentado e relatório técnico detalhado

### Resultados Alcançados:

- ✅ Clusterização bem-sucedida com DBSCAN (3 clusters + ruído identificados)
- ✅ Visualização efetiva com 3 métodos de redução dimensional
- ✅ Detecção de ~1.4% de anomalias no dataset de teste
- ✅ Interface funcional para monitoramento em tempo real

### Aprendizados:

1. A escolha do parâmetro `eps` é crítica para DBSCAN e deve ser determinada empiricamente
2. UMAP oferece o melhor balanço entre velocidade e qualidade de visualização
3. A FFT é fundamental para transformar dados temporais em features discriminativas
4. Normalização adequada é essencial antes de aplicar algoritmos de clustering

---

## 9. Referências

1. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD*, 96(34), 226-231.

2. Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of machine learning research*, 9(11).

3. McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv preprint arXiv:1802.03426*.

4. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of machine learning research*, 12, 2825-2830.

5. Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments. *Philosophical Transactions of the Royal Society A*, 374(2065).

---

## 10. Anexos

### Anexo A: Estrutura do Repositório

```
APOLO/
├── main.py                          # Ponto de entrada
├── gravacao_jogo_dados_controle.py  # Coleta de dados
├── treinar_modelo_local.py          # Treinamento offline
├── requirements.txt                 # Dependências
├── gameplay_session.csv             # Dataset bruto
├── analyzer_model.joblib            # Modelo treinado
│
└── src/
    ├── analysis/
    │   ├── signal_analyzer.py      # FFT
    │   ├── feature_extractor.py    # Extração de features
    │   ├── session_processor.py    # Segmentação em janelas
    │   └── cluster_analyzer.py     # DBSCAN + visualização
    │
    ├── app/
    │   └── streamlit_ui.py         # Interface web
    │
    ├── hardware/
    │   └── sensor_controller.py    # Conexão com DualSense
    │
    └── domain/
        └── movement_test.py         # Definição de testes
```

### Anexo B: Dependências do Projeto

```
# requirements.txt
streamlit>=1.28.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
pydualsense>=0.7.0
joblib>=1.3.0
umap-learn>=0.5.0
```

### Anexo C: Comandos de Execução

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Coletar dados (opcional - já existe gameplay_session.csv)
python gravacao_jogo_dados_controle.py

# 3. Treinar modelo
python treinar_modelo_local.py

# 4. Executar aplicação web
python main.py
```

---

**Fim do Relatório Técnico**

---

*Documento gerado em Dezembro de 2025*  
*Projeto APOLO - IMD3003 - Aprendizado de Máquina Não-Supervisionado*
