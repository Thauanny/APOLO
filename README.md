# APOLO

### Análise Parkinsoniana por Oscilação de Longa Ocorrência

Uma aplicação web, desenvolvida em Python, para a análise de sinais de movimento através de sensores de controle, com foco na deteção de tremores de repouso.

-----

## 🎯 Sobre o Projeto

**APOLO** é uma ferramenta de prova de conceito concebida para explorar a viabilidade de usar hardware de consumo (controles de videojogos com sensores de movimento) para capturar e analisar padrões de movimento humano. A aplicação foca-se especificamente na análise de frequência de sinais do acelerómetro para identificar oscilações rítmicas na faixa de **4 a 8 Hz**, que é a "assinatura" característica do tremor de repouso associado à doença de Parkinson.

O projeto utiliza uma interface web interativa, construída com Streamlit, para proporcionar uma experiência de utilizador simples e visualmente informativa.

## ✨ Funcionalidades

  * **Conexão com Controle:** Interface direta com controles Sony DualSense via Bluetooth.
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
  * **PyDualSense:** Para a comunicação com o controle Sony DualSense.
  * **NumPy & SciPy:** Para a computação numérica e a análise de sinal (FFT).
  * **Matplotlib:** Para a geração dos gráficos.

## 🚀 Começando

Para executar este projeto localmente, siga os passos abaixo.

### Pré-requisitos

  * Python 3.10 ou superior
  * Git
  * Um controle Sony DualSense (PS5) ou DualShock 4 (PS4) conectado ao seu computador via Bluetooth.

### Instalação

1.  **Clone o repositório:**

    ```sh
    git clone https://github.com/seu-usuario/APOLO.git
    cd APOLO
    ```

2.  **Crie e ative um ambiente virtual:**

    ```sh
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # Linux / macOS
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Instale as dependências:**

    ```sh
    pip install -r requirements.txt
    ```

## 🏃 Como Executar

Com o ambiente virtual ativado e o controle conectado:

1.  Execute o `main.py` para lançar a aplicação:
    ```sh
    python main.py
    ```
2.  Uma nova aba abrir-se-á automaticamente no seu navegador web.
3.  Na interface web, clique em "Conectar ao Controle" e siga as instruções para realizar um teste.

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
