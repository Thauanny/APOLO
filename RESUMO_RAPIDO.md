# ⚡ APOLO - Resumo Executivo Rápido

## O que é APOLO?

**Sistema de detecção de anomalias de movimento usando sensores de controle PS5**

Objetivo: Identificar padrões anormais de movimento (especialmente tremor de repouso) comparando dados pessoais com um baseline treinado.

---

## 🎮 Fluxo Principal em 3 Etapas

### **Etapa 1: Coleta de Dados** 📱
```
Você conecta um controle DualSense (PS5) ao PC
    ↓
Script "gravacao_jogo_dados_controle.py" grava:
    ├─ Acelerómetro (X, Y, Z)
    ├─ Giroscópio (pitch, yaw, roll)
    └─ Botões pressionados
    
Resultado: gameplay_session.csv (~1000 linhas de dados brutos)
```

### **Etapa 2: Processamento & Extração de Features** 🔬
```
gameplay_session.csv (dados brutos)
    ↓
SessionProcessor (janelas deslizantes de 2 seg, 50% overlap)
    ↓
FeatureExtractor (FFT + estatísticas)
    ↓
4 Features por teste:
    ├─ peak_freq: frequência dominante (Hz)
    ├─ tremor_power: energia na faixa 4-8 Hz
    ├─ total_power: energia total
    └─ tremor_index: proporção de tremor

Resultado: ~1927 linhas de features (cada 2 seg = 1 feature vector)
```

### **Etapa 3: Treinamento & Detecção** 🤖
```
Features DataFrame
    ↓
ClusterAnalyzer (DBSCAN)
    ├─ Agrupa dados similares
    ├─ Identifica "normalidade" (clusters encontrados)
    └─ Salva em: analyzer_model.joblib

Uso posterior:
    Novo teste (10 seg) → FFT → Features → Comparar com modelo
    ├─ ✅ NORMAL: Dentro dos padrões treinados
    └─ 🚨 ANOMALIA: Fora dos padrões (possível tremor)
```

---

## 📊 Papel da FFT

**FFT = Transformada Rápida de Fourier**

Converte som/movimento **do tempo para frequência**

```
Entrada: 10 segundos de aceleração em X
┌─────────────────────────────────────────────────────┐
│ Aceleração (m/s²) ao longo do tempo                 │
│  Looks like: /\/\/\/\/\/\/\/\ (oscilação rítmica)   │
└─────────────────────────────────────────────────────┘
                    ↓ FFT
                    
Saída: Espectro de frequência
┌─────────────────────────────────────────────────────┐
│ Amplitude em cada frequência (Hz)                    │
│         ╱╲                                           │
│        ╱  ╲  ← Pico em 5.2 Hz (TREMOR!)             │
│───────╱────╲──────────────────────────────────→ Hz  │
│    0  4  5  6  7  8                                 │
│    ↑                                                 │
│    Zona de Tremor Parkinson (4-8 Hz)               │
└─────────────────────────────────────────────────────┘
```

**Por que importa?** O tremor de Parkinson tem "assinatura" específica em 4-8 Hz. FFT permite identificá-la.

---

## 🧠 DBSCAN - O Algoritmo de Clustering

**Agrupa dados similares e identifica outliers**

```
Dado: 1927 pontos em espaço de 7 dimensões

DBSCAN descobre:
├─ Cluster 0: 1500 pontos (maioria, padrão normal 1)
├─ Cluster 1: 300 pontos (variação normal)
├─ Cluster 2: 127 pontos (edge cases, ainda normal)
└─ Ruído (-1): pontos isolados (anomalias)

Definição de "normalidade": TODOS os clusters (0, 1, 2)
Definição de "anomalia": Pontos fora desses clusters
```

**Parâmetros:**
- `eps = 5.0`: Quão perto dois pontos precisam estar para pertencer ao mesmo cluster
- `min_samples = 14`: Mínimo de vizinhos próximos para ser considerado cluster denso

---

## 📈 Visualizações e Seu Significado

### **1. Sinal no Tempo** (gráfico 1 em testes)
```
Mostra: Aceleração bruta ao longo de 10 seg
├─ Tremor = oscilação rítmica e regular
├─ Normal = movimento caótico ou estável
└─ Para o usuário: confirmar visualmente
```

### **2. Espectro FFT** (gráfico 2 em testes)
```
Mostra: Quais frequências têm mais energia
├─ Pico em 4-8 Hz = tremor detectado
├─ Sem picos claros = sem tremor característico
└─ Para o usuário: Confirmar faixa de frequência
```

### **3. Clusters 2D** (na análise de sessão)
```
Mostra: Visualização dos 1927 pontos de features
Método: Reduz de 7D para 2D usando:
    ├─ PCA: Rápido, preserva estrutura global
    ├─ t-SNE: Lento, destaca agrupamentos locais
    └─ UMAP: Rápido, não-linear

Interpretação:
├─ Cores diferentes = clusters diferentes
├─ Cores bem separadas = modelo bom
├─ Cores misturadas = parâmetros precisam ajuste
└─ Para o usuário: Visualizar separação entre padrões
```

### **4. K-Distance Graph** (ferramentas)
```
Mostra: Distância ao k-ésimo vizinho
├─ Procura: "Cotovelo" na curva
├─ Valor naquele ponto = eps ideal para DBSCAN
└─ Para o usuário: Otimizar parâmetros manualmente
```

---

## 🚀 Como Usar (Passo a Passo)

### **Primeira Vez: Treinar Modelo Pessoal**

```bash
# 1. Gerar dados de baseline (quando você se sente bem)
python gravacao_jogo_dados_controle.py
    # Conecta ao DualSense
    # Pressione [Enter]
    # Jogue ~30 minutos normalmente
    # Pressione [Ctrl+C] para parar
    # Resulta: gameplay_session.csv (1027 linhas)

# 2. Treinar o modelo
python treinar_modelo_local.py
    # Lê gameplay_session.csv
    # Processa e extrai 1927 features
    # Treina DBSCAN
    # Salva em: analyzer_model.joblib
    # (~2 minutos)
```

### **Uso Diário: Monitorização**

```bash
# 1. Abrir a app web
python main.py
    # Abre em http://localhost:8501

# 2. Modo "Monitorização" (tempo real)
    # "Conectar ao Controle" (DualSense)
    # "Iniciar Teste" (10 segundos)
    # App mostra: Gráficos + Resultado (✅ ou 🚨)

# 3. Modo "Análise de Sessão" (histórico)
    # Carregar gameplay_session.csv (ou outro)
    # Visualizar 1927 pontos em 2D
    # Escolher: PCA / t-SNE / UMAP
    # Interpretar clusters
```

---

## 🔑 Conceitos-Chave

| Termo | O que é | Por que importa |
|-------|---------|-----------------|
| **FFT** | Transforma sinal temporal em frequencial | Identifica frequências de tremor |
| **DBSCAN** | Agrupa dados por densidade | Identifica padrões e anomalias |
| **Baseline** | Seus dados pessoais de treino | Define o que é "normal" para você |
| **Features** | 7 números que descrevem 2 seg de movimento | Reduzem 200 pontos para 7 números |
| **Cluster** | Grupo de dados similares | Definem regiões de "normalidade" |
| **Anomalia** | Ponto fora dos clusters | Possível tremor ou padrão anormal |
| **eps** | Raio de vizinhança DBSCAN | Define tamanho dos clusters |
| **Tremor Index** | tremor_power / total_power | Percentual de tremor no sinal |

---

## 📊 Arquivo mais importante: `gameplay_session.csv`

**Estrutura:**
```
timestamp | accel_x | accel_y | accel_z | gyro_x | gyro_y | gyro_z | R1 | L1 | ...
0.001     | -0.50   | 0.20    | 9.80    | 0.01   | 0.02   | 0.03   | 0  | 0  | ...
0.011     | -0.48   | 0.21    | 9.79    | 0.015  | 0.025  | 0.031  | 0  | 0  | ...
0.021     | -0.45   | 0.22    | 9.78    | 0.020  | 0.028  | 0.029  | 0  | 1  | ...
```

**Uso:**
1. **Treino:** SessionProcessor divide em 1927 janelas → extrai features
2. **Análise:** Carregue em "Análise de Sessão" para visualizar clusters
3. **Comparação:** Use para validar novo baseline

---

## 🎯 Interpretações Práticas

### **Cenário 1: Teste de Monitorização mostra ✅ NORMAL**
```
Significado: Padrão de movimento dentro do baseline treinado
Causa provável: Você está bem (sem anomalias)
Ação: Continuar usando normalmente
```

### **Cenário 2: Teste de Monitorização mostra 🚨 ANOMALIA**
```
Significado: Padrão diferente do baseline
Causas possíveis:
    ├─ Tremor mais forte que o usual
    ├─ Padrão diferente de movimento
    └─ Dia atípico (estresse, cansaço, etc.)
Ação: Verificar com profissional se apropriado
```

### **Cenário 3: Visualização de clusters bem separados**
```
Significado: Modelo está funcionando bem
Causa: Seus padrões de movimento são consistentes
Ação: Modelo pronto para uso confiável
```

### **Cenário 4: Visualização de clusters misturados**
```
Significado: Parâmetros DBSCAN não ideais
Ação: Usar K-Distance graph para encontrar melhor eps
```

---

## ⚠️ Limitações Importantes

```
✓ O que APOLO faz bem:
    ├─ Detectar variações no seu padrão pessoal
    ├─ Quantificar tremor (frequência e amplitude)
    └─ Visualizar dados de movimento

✗ O que APOLO NÃO faz:
    ├─ Diagnosticar Parkinson (requer médico)
    ├─ Ser 100% preciso (sensores de console vs médicos)
    └─ Substituir avaliação clínica

⚖️ É uma ferramenta de exploração/monitorização, não diagnóstico
```

---

## 📂 Arquivos Principais do Projeto

```
config.py                          ← Configuração centralizada (DBSCAN, tremor, etc.)
gravacao_jogo_dados_controle.py    ← Gera gameplay_session.csv
treinar_modelo_local.py            ← Treina analyzer_model.joblib
main.py                            ← Lança a interface Streamlit

src/analysis/
    ├─ signal_analyzer.py          ← FFT (coração da análise)
    ├─ feature_extractor.py        ← 4 features por teste
    ├─ session_processor.py        ← Divide em janelas
    └─ cluster_analyzer.py         ← DBSCAN + redução dimensional (Singleton)

src/app/
    └─ streamlit_ui.py             ← Interface web
```

---

## 🚀 Próximas Melhorias Possíveis

```
Curto Prazo:
    ├─ Salvar histórico de testes
    ├─ Gráficos de evolução ao longo do tempo
    └─ Alertas automáticos

Médio Prazo:
    ├─ Analisar eixos Y e Z além de X
    ├─ Usar dados do giroscópio
    └─ Mais tipos de movimento (tapping, etc.)

Longo Prazo:
    ├─ Deep Learning em vez de DBSCAN
    ├─ Sincronizar com aplicativo móvel
    └─ Banco de dados em nuvem
```

---

## 💡 Dúvidas Comuns Respondidas

**P: E se não tenho tremor?**  
R: A app funciona! Será apenas com tremor_index baixo. Útil para monitorar variações.

**P: Preciso treinar com dados de outras pessoas?**  
R: Não. Cada pessoa tem padrão único. Sempre treine com seus próprios dados.

**P: Posso usar outro controle?**  
R: Sim, qualquer compatível com `pydualsense` (DualSense, DualShock 4).

**P: A app é um dispositivo médico?**  
R: Não. É uma prova de conceito. Nunca use para autodiagnóstico.

**P: Como melhorar a precisão?**  
R: Mais dados de treino (60+ minutos), otimizar eps via K-Distance graph.

---

## 📞 Estrutura Resumida

```
┌──────────────────────────────────────────────────────────────┐
│                    FLUXO COMPLETO                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Hardware: DualSense (PS5) ←→ sensor_controller.py          │
│                │                                            │
│                ↓                                            │
│  Gravação: gameplay_session.csv (dados brutos)             │
│                │                                            │
│                ↓                                            │
│  Processamento: SessionProcessor (divide em janelas)        │
│                │                                            │
│                ↓                                            │
│  Extração: feature_extractor.py (FFT + estatísticas)       │
│                │                                            │
│                ↓                                            │
│  Treino: ClusterAnalyzer (DBSCAN)                          │
│                │                                            │
│                ↓                                            │
│  Modelo: analyzer_model.joblib (salvo)                     │
│                │                                            │
│                ↓                                            │
│  Predição: Novo teste → Features → Comparar com modelo     │
│                │                                            │
│                ↓                                            │
│  Resultado: ✅ NORMAL ou 🚨 ANOMALIA                       │
│                │                                            │
│                ↓                                            │
│  UI: Streamlit (gráficos + visualizações)                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

**Entendeu o projeto?** Se tiver dúvidas específicas, é só perguntar! 🚀
