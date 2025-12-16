# APOLO

### Análise Parkinsoniana por Oscilação de Longa Ocorrência

Uma aplicação web, desenvolvida em Python, para a análise de sinais de movimento através de sensores de controle, com foco na deteção de tremores de repouso.

-----

## 🎯 Sobre o Projeto

**APOLO** é uma ferramenta de prova de conceito concebida para explorar a viabilidade de usar hardware de consumo (controles de videojogos com sensores de movimento) para capturar e analisar padrões de movimento humano. A aplicação foca-se especificamente na análise de frequência de sinais do acelerómetro para identificar oscilações rítmicas na faixa de **4 a 8 Hz**, que é a "assinatura" característica do tremor de repouso associado à doença de Parkinson.

O projeto utiliza uma interface web interativa, construída com Streamlit, para proporcionar uma experiência de utilizador simples e visualmente informativa.

## ✨ Funcionalidades

  * **Conexão com Controle:** Interface direta com controles Sony DualSense via cabo USB.
  * **Recolha de Dados em Tempo Real:** Captura de dados do acelerómetro e giroscópio durante testes com duração definida.
  * **Análise de Sinal Avançada:** Utiliza a Transformada Rápida de Fourier (FFT) com a biblioteca `SciPy` para decompor o sinal de movimento no domínio da frequência.
  * **Deteção Focada:** Isola e analisa a banda de frequência de 4-8 Hz para encontrar o pico de oscilação mais dominante, característico de tremores de repouso.
  * **Visualização Interativa:** Apresenta os resultados numa interface web limpa, com gráficos do sinal no tempo e da análise de frequência.
  * **Arquitetura Robusta:** Desenvolvido com uma arquitetura de camadas (Apresentação, Aplicação, Hardware, Análise) e princípios de Programação Orientada a Objetos (POO).

## 🔬 Como Funciona

1.  **Captura:** A aplicação estabelece uma conexão com o controle DualSense.
2.  **Análise:** Durante um teste, os dados do acelerómetro são recolhidos. Este sinal (domínio do tempo) é então processado usando a FFT para o converter para o domínio da frequência.
3.  **Deteção:** O algoritmo filtra o resultado da FFT, focando-se apenas na janela de 4 a 8 Hz, e identifica a frequência com a maior amplitude (o pico de energia).
4.  **Visualização:** Os resultados, incluindo o sinal bruto e o espectro de frequência com o pico destacado, são renderizados na interface web do Streamlit.

## 🛠️ Tecnologias Utilizadas

  * **Python 3.10+**
  * **Streamlit:** Para a construção da interface web interativa.
  * **PyDualSense:** Para a comunicação com o controle Sony DualSense com HIDAPI
  * **NumPy & SciPy:** Para a computação numérica e a análise de sinal (FFT).
  * **Matplotlib:** Para a geração dos gráficos.

## 🚀 Começando

Para executar este projeto localmente, siga os passos abaixo.

### Pré-requisitos

  * Python 3.10 ou superior
  * Git
  * **hidapi** (biblioteca de sistema - veja instruções de instalação abaixo)
  * Um controle Sony DualSense (PS5) ou DualShock 4 (PS4) **conectado via cabo USB**

### Instalação

1.  **Clone o repositório:**

    ```sh
    git clone https://github.com/seu-usuario/APOLO.git
    cd APOLO
    ```

2.  **Instale as dependências de sistema:**

    O projeto usa `pydualsense` que depende da biblioteca `hidapi`. O procedimento varia conforme o sistema operacional:

    **macOS:**
    ```sh
    brew install hidapi
    ```

    **Linux (Debian/Ubuntu):**
    ```sh
    sudo apt-get install libhidapi-dev
    ```

    **Windows:**
    - Baixe o instalador de `hidapi` em: https://github.com/libusb/hidapi/releases
    - Ou use uma distribuição pré-compilada com: `pip install hidapi`
    - Se encontrar problemas, reinstale os Visual C++ Build Tools

3.  **Crie e ative um ambiente virtual:**

    ```sh
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # Linux / macOS
    python3 -m venv .venv
    source .venv/bin/activate
    ```

4.  **Instale as dependências Python:**

    ```sh
    pip install -r requirements.txt
    ```

## 🏃 Como Executar

**IMPORTANTE:** O controle DEVE estar conectado via **cabo USB** antes de executar a aplicação.

### macOS e Linux

Execute o seguinte comando na raiz do projeto:

```bash
source activate_hidapi.sh
python main.py
```

### Windows

No Windows, `hidapi` funciona automaticamente. Basta ativar o ambiente virtual e executar:

```cmd
.venv\Scripts\activate
python main.py
```

> **Nota:** Windows não precisa de configurações especiais de variáveis de ambiente. O `hidapi` é instalado automaticamente via pip e funciona nativamente.

---

**Em qualquer sistema, a aplicação abrirá automaticamente em `http://localhost:8502`**

**Próximas Etapas:**
1. Uma nova aba abrir-se-á automaticamente no seu navegador web
2. Na interface web, clique em **"Conectar ao Controle"**
3. Siga as instruções para realizar um teste

## 🔧 Solução de Problemas

### OSError: Could not find any hidapi library

**Solução:**

Certifique-se de que está usando o script de ativação:

```bash
source activate_hidapi.sh && python main.py
```

Não execute diretamente:
```bash
python main.py  # ❌ Vai dar erro!
```

O script `activate_hidapi.sh` configura a variável de ambiente `DYLD_LIBRARY_PATH` necessária.

### Controle não é detectado

1. **Certifique-se de que o controle está conectado via cabo USB** (Bluetooth não é suportado)
2. Verifique se o cabo USB está bem conectado
3. Tente um cabo USB diferente
4. Reinicie o controle e a aplicação
5. Se o problema persistir, verifique as permissões de acesso a dispositivos USB

### Streamlit não abre automaticamente

- Se a aba não abrir, acesse manualmente: `http://localhost:8502`

## 📚 Estrutura do Projeto

```
APOLO/
├── src/
│   ├── analysis/          # Análise de sinais e detecção
│   ├── app/              # Interface Streamlit
│   ├── domain/           # Testes de movimento
│   ├── hardware/         # Controle do sensor
│   └── utils/            # Utilitários e gráficos
├── main.py               # Entrada principal da aplicação
├── requirements.txt      # Dependências Python
├── activate_hidapi.sh    # Script de ativação (macOS)
└── README.md            # Este arquivo
```

## 📖 Como Usar (Guia Completo)

### **Primeira Vez: Treinar Seu Modelo Pessoal**

Antes de usar a aplicação, você precisa treinar um modelo com seus dados pessoais de baseline:

#### Passo 1: Coletar dados de baseline
```bash
python gravacao_jogo_dados_controle.py
```
**O que faz:**
- Conecta ao seu controle DualSense
- Pressione [Enter] para começar
- Jogue normalmente ~30 minutos (quando você se sente bem)
- Pressione [Ctrl+C] para parar
- Salva em: `gameplay_session.csv` (~1027 linhas de dados)

#### Passo 2: Treinar o modelo
```bash
python treinar_modelo_local.py
```
**O que faz:**
- Lê `gameplay_session.csv`
- Processa dados e extrai 1927 features (7 features por janela de 2 seg)
- Treina o algoritmo DBSCAN
- Salva o modelo em: `analyzer_model.joblib`
- ⏱️ Tempo: ~2 minutos

### **Uso Diário: Monitorização**

#### Opção 1: Teste em Tempo Real (Monitorização)
```bash
python main.py
```
**Na interface web:**
1. Clique em **"Conectar ao Controle"** e selecione seu DualSense
2. Clique em **"Iniciar Teste de Monitorização"**
3. A aplicação coletará dados por 10 segundos
4. Resultado: 
   - ✅ **NORMAL** = Padrão dentro do seu baseline
   - 🚨 **ANOMALIA** = Padrão diferente do seu baseline

#### Opção 2: Análise de Sessão (Histórico)
```bash
python main.py
```
**Na interface web:**
1. Clique em **"Análise de Sessão"**
2. Carregue um arquivo `gameplay_session.csv`
3. Escolha método de visualização:
   - **PCA** - Rápido, preserva estrutura global
   - **t-SNE** - Lento, destaca agrupamentos locais
   - **UMAP** - Rápido, análise não-linear
4. Interprete os clusters:
   - Cores bem separadas = modelo funcionando bem
   - Cores misturadas = parâmetros precisam ajuste

## 🔍 Entendendo os Resultados

### **O que significam os resultados?**

| Resultado | Significado | O que fazer |
|-----------|------------|------------|
| ✅ NORMAL | Seu padrão está dentro do baseline | Continue normalmente |
| 🚨 ANOMALIA | Padrão diferente do baseline | Verificar com profissional se apropriado |
| Clusters bem separados | Modelo está funcionando bem | Usar com confiança |
| Clusters misturados | Parâmetros não ideais | Otimizar via K-Distance graph |

### **Como funciona a detecção?**

1. **Captura:** Coleta dados do acelerómetro por 10 segundos
2. **FFT:** Transforma dados temporais em frequências
3. **Features:** Extrai 7 características (frequência dominante, energia, etc.)
4. **Comparação:** Compara com seu modelo treinado (DBSCAN)
5. **Resultado:** Normal ou Anomalia

## ⚠️ Aviso Importante

**Este projeto é uma ferramenta de exploração e uma prova de conceito. NÃO É UM DISPOSITIVO MÉDICO.**

Os resultados gerados por esta aplicação não devem, em nenhuma circunstância, ser usados para autodiagnóstico ou para tomar decisões clínicas. A precisão dos sensores de um controle de videojogos não é calibrada para fins médicos. Para qualquer questão de saúde, **consulte sempre um profissional de saúde qualificado**.

<!-- ## 🗺️ Próximos Passos

  * [ ] Implementar um sistema para guardar e carregar os resultados dos testes.
  * [ ] Criar perfis de utilizador para acompanhar a evolução ao longo do tempo.
  * [ ] Adicionar a análise dos outros eixos do acelerómetro e dos dados do giroscópio.
  * [ ] Permitir a comparação lado a lado de diferentes testes. -->

## 📄 Licença

Este projeto é distribuído sob uma licença proprietária.

**Copyright (c) 2025 Thauanny Kyssy Ramos Pereira**

**Todos os Direitos Reservados.**

A utilização, reprodução, modificação ou distribuição deste software ou de qualquer parte dele, sem a permissão explícita por escrito do detentor dos direitos de autor, é estritamente proibida.
