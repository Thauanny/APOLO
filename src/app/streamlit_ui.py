# Copyright (c) 2025 Thauanny Kyssy Ramos Pereira. Todos os Direitos Reservados.
#
# Este software é propriedade confidencial e proprietária de Thauanny Kyssy Ramos Pereira.
# A utilização, cópia ou divulgação deste ficheiro só é permitida de acordo
# com os termos de um contrato de licença celebrado com o autor.

import streamlit as st
import time
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from src.hardware.sensor_controller import SensorController
from src.domain.movement_test import MovementTest
from src.analysis.signal_analyzer import SignalAnalyzer
from src.analysis.feature_extractor import extract_features
from src.utils.plotter import plot_test_results
from src.analysis.cluster_analyzer import ClusterAnalyzer
from src.analysis.session_processor import SessionProcessor
from sklearn.decomposition import PCA

MODEL_PATH = "analyzer_model.joblib"

@st.cache_data
def compute_k_distance_graph(features_df: pd.DataFrame, k: int):
    """Calcula e guarda em cache os dados para o gráfico K-Distance."""
    print("INFO: (Terminal) A calcular o gráfico K-Distance...")
    distances = ClusterAnalyzer.calculate_k_distance_graph(features_df, k=k)
    print("INFO: (Terminal) Cálculo do K-Distance concluído.")
    return distances

class StreamlitApp:
    def __init__(self):
        st.set_page_config(page_title="APOLO", layout="wide")
        self._initialize_session_state()
        
        self.TESTS = {
            "Repouso na Mão": MovementTest(name="Repouso na Mão", instructions="Segure o controle parado na sua mão, apoiado na perna.", duration_seconds=10),
            "Teste de Tapping Rápido": MovementTest(name="Teste de Tapping Rápido", instructions="Pressione o botão 'R1' o mais rápido que conseguir.", duration_seconds=10),
            "Teste de Tremor Simulado": MovementTest(name="Teste de Tremor Simulado", instructions="Tente oscilar a mão num ritmo constante (use um metrónomo a 360bpm).", duration_seconds=10),
        }

    def _initialize_session_state(self):
        if 'controller' not in st.session_state: st.session_state.controller = None
        if 'analyzer' not in st.session_state: st.session_state.analyzer = None
        if 'model_loaded' not in st.session_state: st.session_state.model_loaded = False
        if 'last_test_result' not in st.session_state: st.session_state.last_test_result = None

    def run(self):
        st.sidebar.title("APOLO")
        mode = st.sidebar.radio("Navegação", ["Monitorização", "Análise de Sessão de Jogo", "Ferramentas de Análise"])

        if not st.session_state.model_loaded:
            try:
                st.session_state.analyzer = ClusterAnalyzer.load_model(MODEL_PATH)
                st.session_state.model_loaded = True
            except FileNotFoundError:
                st.session_state.model_loaded = False

        if mode == "Monitorização":
            self._render_monitoring_view()
        elif mode == "Análise de Sessão de Jogo":
            self._render_analysis_view()
        elif mode == "Ferramentas de Análise":
            self._render_tools_view()

    def _render_tools_view(self):
        st.title("🛠️ Ferramentas de Análise - Gráfico K-Distance")
        st.info("Faça o upload de um dataset de **features** (pós-processamento) para explorar a sua estrutura de densidade e ajudar a encontrar um bom `eps` para um futuro treino offline.")
        
        uploaded_file = st.file_uploader("Escolha um ficheiro CSV de features para explorar", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            with st.sidebar:
                st.header("Configuração da Ferramenta")
                all_features = df.drop(columns=['label'], errors='ignore').columns.tolist()
                features_to_use = st.multiselect("Selecione as features para a análise:", options=all_features, default=all_features)
                min_samples_for_k = st.slider("Amostras Mínimas (k) para o gráfico:", 1, 20, 10, 1)

            if not features_to_use:
                st.warning("Selecione pelo menos uma feature.")
                return
                
            features_df = df[features_to_use]

            with st.spinner("A calcular gráfico K-Distance..."):
                distances = compute_k_distance_graph(features_df, k=min_samples_for_k)
            
            fig_k, ax_k = plt.subplots()
            ax_k.plot(distances)
            ax_k.set_title(f"Gráfico K-Distance (para k = {min_samples_for_k})")
            ax_k.set_xlabel("Pontos de Dados (ordenados por distância)")
            ax_k.set_ylabel(f"Distância ao {min_samples_for_k}º Vizinho")
            ax_k.grid(True)
            plot_col, _ = st.columns([0.7, 0.3])
            with plot_col:
                st.pyplot(fig_k)
            st.success("Analise o 'cotovelo' no gráfico para estimar o melhor `eps` para usar no seu script de treino offline.")

    def _render_analysis_view(self):
        st.title("📊 Análise de Sessão de Jogo Gravada")
        st.info(f"Faça o upload de um ficheiro de dados brutos (ex: `gameplay_session.csv`) para extrair as features e visualizar os clusters com o modelo pré-treinado (`{MODEL_PATH}`).")
        
        if not st.session_state.model_loaded:
            st.error(f"ERRO: Modelo '{MODEL_PATH}' não encontrado.", icon="⛔")
            st.info("Execute o script 'treinar_modelo_local.py' primeiro para gerar o modelo.")
            return

        with st.sidebar:
            st.header("Modelo Carregado")
            analyzer: ClusterAnalyzer = st.session_state.analyzer
            st.metric(label="Epsilon (eps) do Modelo", value=f"{analyzer.eps}")
            st.metric(label="Amostras Mínimas do Modelo", value=f"{analyzer.min_samples}")
        
        uploaded_file = st.file_uploader("Escolha um ficheiro CSV de sessão de jogo", type="csv")
        
        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            
            with st.spinner("A processar a sessão e a extrair features... Isto pode demorar."):
                processor = SessionProcessor()
                features_df = processor.process_session_df(raw_df)

            if features_df.empty:
                st.warning("Não foi possível extrair features do arquivo fornecido.")
                return

            st.success(f"Sessão processada! Foram extraídas features de {len(features_df)} janelas.")
            
            with st.spinner("A aplicar o modelo pré-treinado..."):
                predicted_labels = st.session_state.analyzer.predict_clusters(features_df)
            
            df_display = features_df.copy()
            df_display['cluster'] = predicted_labels

            st.header("Resultados da Clusterização")
            st.metric("Clusters de 'Normalidade' Encontrados", len(set(predicted_labels) - {-1}))
            st.metric("Janelas Anómalas Detetadas", list(predicted_labels).count(-1))

            if features_df.shape[1] < 2:
                st.warning("A visualização PCA requer pelo menos 2 features.")
            else:
                scaled_features = ClusterAnalyzer._scale_features(features_df)
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(scaled_features)
                fig_pca, ax_pca = plt.subplots(figsize=(10, 7))
                for cluster_id in sorted(np.unique(predicted_labels)):
                    label = 'Anomalia (Ruído)' if cluster_id == -1 else f'Cluster {cluster_id}'
                    color = 'red' if cluster_id == -1 else f'C{cluster_id}'
                    marker = 'x' if cluster_id == -1 else 'o'
                    indices = np.where(predicted_labels == cluster_id)
                    ax_pca.scatter(principal_components[indices, 0], principal_components[indices, 1],
                               label=label, c=color, marker=marker, s=100, alpha=0.7)
                ax_pca.set_title("Visualização dos Clusters da Sessão de Jogo")
                ax_pca.legend()
                ax_pca.grid(True)
                plot_col, _ = st.columns([0.7, 0.3])
                with plot_col:
                    st.pyplot(fig_pca)
            
            st.write("### Tabela de Janelas de Análise com Clusters:", df_display)

    def _render_monitoring_view(self):
        st.title("🕵️‍♂️ Monitorização de Anomalias Motoras")
        if not st.session_state.model_loaded:
            st.error(f"ERRO: Modelo '{MODEL_PATH}' não encontrado.", icon="⛔")
            st.info("Execute o script 'treinar_modelo_local.py' primeiro para gerar o modelo.")
            return
        
        st.success("Modelo de deteção de anomalias carregado e pronto para uso.", icon="🤖")
        with st.sidebar:
            self._render_connection_controls()
            st.divider()
            st.header("Parâmetros do Modelo Ativo")
            analyzer: ClusterAnalyzer = st.session_state.analyzer
            st.metric(label="Epsilon (eps)", value=f"{analyzer.eps}")
            st.metric(label="Amostras Mínimas", value=f"{analyzer.min_samples}")
            st.divider()
            st.header("Realizar Novo Teste")
            selected_test_name = st.selectbox("Escolha um teste para monitorizar:", options=list(self.TESTS.keys()), disabled=(st.session_state.controller is None))
            if st.button("🚀 Iniciar Teste de Monitorização", type="primary", disabled=(st.session_state.controller is None)):
                selected_test = self.TESTS[selected_test_name]
                self._run_test_logic(selected_test)
        self._render_monitoring_results()

    def _render_connection_controls(self):
        if st.session_state.controller is None:
            if st.button("🔌 Conectar ao Controle"):
                with st.spinner("Procurando..."):
                    try:
                        st.session_state.controller = SensorController()
                        st.rerun()
                    except ConnectionError as e:
                        st.error(f"Falha na conexão: {e}")
        else:
            st.success("✅ Controlador Conectado")
            if st.button("🔌 Desconectar"):
                st.session_state.controller.close()
                st.session_state.controller = None
                st.rerun()

    def _render_monitoring_results(self):
        st.header("Resultado da Análise")
        last_result = st.session_state.last_test_result
        if last_result is None:
            st.info("Aguardando a execução de um teste.")
            return
        st.write(f"### Análise para: {last_result['name']}")
        features = extract_features(last_result)
        is_anomalous = st.session_state.analyzer.predict_is_anomalous(features)
        if is_anomalous:
            st.error("🚨 ALERTA: Anomalia detectada no padrão de movimento!", icon="🚨")
        else:
            st.success("✅ Padrão de movimento dentro da normalidade.", icon="✅")
        if "Repouso" in last_result['name'] or "Tremor" in last_result['name']:
            fig = plot_test_results(time_axis=last_result['timestamps'], sensor_data=last_result['readings'], fft_results=last_result['fft_results'], test_name=last_result['name'])
            plot_col, _ = st.columns([0.7, 0.3])
            with plot_col:
                st.pyplot(fig)
        elif "Tapping" in last_result['name']:
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total de Cliques", f"{features['tap_count']} cliques")
            with col2: st.metric("Frequência Média", f"{features['tap_freq']:.2f} cliques/s")
            with col3: st.metric("Índice de Irregularidade (DP)", f"{features['tap_interval_std']:.4f} s")

    def _run_test_logic(self, test: MovementTest):
        result_data = None
        with st.spinner(f"Executando '{test.name}'..."):
            if "Repouso" in test.name or "Tremor" in test.name:
                timestamps, readings = [], []
                start_time = time.time()
                while time.time() - start_time < test.duration_seconds:
                    try:
                        readings.append(st.session_state.controller.get_sensors_data()['accel_x'])
                        timestamps.append(time.time() - start_time)
                    except TimeoutError: continue
                    time.sleep(0.01)
                if readings:
                    analyzer = SignalAnalyzer()
                    sample_rate = len(readings) / test.duration_seconds
                    fft_results = analyzer.find_tremor_frequency(readings, sample_rate)
                    result_data = {"name": test.name, "timestamps": timestamps, "readings": readings, "fft_results": fft_results, "sample_rate": sample_rate}
            elif "Tapping" in test.name:
                st.session_state.controller.start_tapping_test()
                time.sleep(test.duration_seconds)
                readings = st.session_state.controller.get_tapping_results()
                result_data = {"name": test.name, "readings": readings, "duration": test.duration_seconds}
        st.session_state.last_test_result = result_data

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()