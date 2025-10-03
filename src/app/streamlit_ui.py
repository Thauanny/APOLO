# Copyright (c) 2025 Thauanny Kyssy Ramos Pereira. Todos os Direitos Reservados.
# ... (cabeçalho de copyright igual)

"""
Este módulo contém a classe que constrói e executa a interface do utilizador
da aplicação APOLO usando o Streamlit.
"""

import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
from src.hardware.sensor_controller import SensorController
from src.domain.movement_test import MovementTest
from src.utils.plotter import plot_test_results
from src.analysis.signal_analyzer import SignalAnalyzer
from src.analysis.feature_extractor import extract_features
from src.analysis.cluster_analyzer import ClusterAnalyzer

class StreamlitApp:
    """
    Encapsula a lógica de renderização da interface gráfica.
    Delega a lógica de negócio e de análise para as classes de controlador.
    """
    def __init__(self):
        st.set_page_config(page_title="APOLO", layout="wide")
        self._initialize_session_state()
        
        self.TESTS = {
            "Teste de Repouso": MovementTest(
                name="Teste de Repouso",
                instructions="Segure o controle parado na sua mão, apoiado na perna.",
                duration_seconds=10
            ),
            "Teste de Tapping Rápido": MovementTest(
                name="Teste de Tapping Rápido",
                instructions="Pressione o botão 'X' o mais rápido que conseguir.",
                duration_seconds=10
            ),
            "Teste de Tremor Simulado": MovementTest(
                name="Teste de Tremor Simulado",
                instructions="Tente oscilar a mão num ritmo constante (use um metrónomo a 360bpm).",
                duration_seconds=10
            )
        }

    def _initialize_session_state(self):
        if 'controller' not in st.session_state:
            st.session_state.controller = None
        if 'test_results' not in st.session_state:
            st.session_state.test_results = []
        if 'dataset_features' not in st.session_state:
            st.session_state.dataset_features = []

    def run(self):
        st.sidebar.title("APOLO")
        mode = st.sidebar.radio(
            "Navegação",
            ["Coleta de Dados", "Análise de Dataset (DBSCAN)"],
            label_visibility="hidden"
        )
        if mode == "Coleta de Dados":
            self._render_data_collection_view()
        elif mode == "Análise de Dataset (DBSCAN)":
            self._render_dbscan_analysis_view()

    def _render_data_collection_view(self):
        st.title("🔬 Coleta e Análise de Movimento")
        with st.sidebar:
            st.header("Configuração de Coleta")
            if st.session_state.controller is None:
                if st.button("🔌 Conectar ao Controle"):
                    with st.spinner("Procurando controlador..."):
                        try:
                            st.session_state.controller = SensorController()
                            st.success("Controlador conectado!")
                            st.rerun()
                        except ConnectionError as e:
                            st.error(f"Falha na conexão: {e}")
            else:
                st.success("✅ Controlador Conectado")
                if st.button("🔌 Desconectar"):
                    st.session_state.controller.close()
                    st.session_state.controller = None
                    st.rerun()

            selected_test_name = st.selectbox("Escolha um teste para realizar:", options=list(self.TESTS.keys()), disabled=(st.session_state.controller is None))
            selected_test = self.TESTS[selected_test_name]
            st.info(f"**Instruções:** {selected_test.instructions}\n\n**Duração:** {selected_test.duration_seconds}s.")

            if st.button("🚀 Iniciar Teste", type="primary", disabled=(st.session_state.controller is None)):
                self._run_test_logic(selected_test)
                st.rerun()

            st.divider()
            st.header("Dataset para Análise")
            if st.session_state.test_results:
                if st.button("➕ Adicionar último teste ao dataset"):
                    last_result = st.session_state.test_results[-1]
                    last_result['label'] = selected_test.name 
                    features = extract_features(last_result)
                    st.session_state.dataset_features.append(features)
                    st.success(f"Adicionado! Total: {len(st.session_state.dataset_features)}")
            if st.session_state.dataset_features:
                df = pd.DataFrame(st.session_state.dataset_features)
                st.download_button(label="📥 Baixar Dataset (CSV)", data=df.to_csv(index=False).encode('utf-8'), file_name='simulated_dataset.csv', mime='text/csv')
                if st.button("🗑️ Resetar Dataset", type="secondary"):
                    st.session_state.dataset_features = []
                    st.success("Dataset em memória foi limpo!")
                    time.sleep(1)
                    st.rerun()
            st.caption(f"Amostras no dataset atual: {len(st.session_state.dataset_features)}")

        st.header("Resultados do Último Teste")
        if not st.session_state.test_results:
            st.info("Aguardando a execução de um teste.")
        else:
            last_result = st.session_state.test_results[-1]
            st.write(f"### Análise para: {last_result['name']}")
            if "Repouso" in last_result['name'] or "Tremor" in last_result['name']:
                _, _, dominant_freq, _ = last_result['fft_results']
                st.metric("Frequência de Pico Detectada (4-8 Hz)", f"{dominant_freq:.2f} Hz")
                fig = plot_test_results(time_axis=last_result['timestamps'], sensor_data=last_result['readings'], fft_results=last_result['fft_results'], test_name=last_result['name'])
                st.pyplot(fig)
            elif "Tapping" in last_result['name']:
                features = extract_features(last_result)
                st.metric("Total de Cliques", f"{features['tap_count']} cliques")
                st.metric("Frequência Média", f"{features['tap_freq']:.2f} cliques/s")
                st.metric("Índice de Irregularidade (DP)", f"{features['tap_interval_std']:.4f} s")

    def _run_test_logic(self, test: MovementTest):
        with st.spinner(f"Executando '{test.name}'..."):
            if "Repouso" in test.name or "Tremor" in test.name:
                timestamps, sensor_readings = [], []
                start_time = time.time()
                while time.time() - start_time < test.duration_seconds:
                    try:
                        data = st.session_state.controller.get_sensors_data()
                        timestamps.append(time.time() - start_time)
                        sensor_readings.append(data['accel_x'])
                    except TimeoutError: continue
                    time.sleep(0.01)
                if sensor_readings:
                    sample_rate = len(timestamps) / test.duration_seconds
                    analyzer = SignalAnalyzer()
                    fft_results = analyzer.find_tremor_frequency(sensor_readings, sample_rate)
                    st.session_state.test_results.append({"name": test.name, "timestamps": timestamps, "readings": sensor_readings, "fft_results": fft_results, "sample_rate": sample_rate})
            elif "Tapping" in test.name:
                st.session_state.controller.start_tapping_test()
                time.sleep(test.duration_seconds)
                tapping_results = st.session_state.controller.get_tapping_results()
                st.session_state.test_results.append({"name": test.name, "readings": tapping_results, "duration": test.duration_seconds})

    def _render_dbscan_analysis_view(self):
        """
        Renderiza a interface do laboratório de análise, agora usando o ClusterAnalyzer.
        """
        st.title("🔬 Laboratório de Análise de Anomalias")
        st.info("Faça o upload de um dataset (como o `simulated_dataset.csv`) para análise.")
        
        uploaded_file = st.file_uploader("Escolha um ficheiro CSV", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            all_features = df.drop(columns=['label'], errors='ignore').columns.tolist()
            with st.sidebar:
                st.header("Configuração da Análise")
                features_to_use = st.multiselect("Selecione as features para a análise:", options=all_features, default=all_features)
                st.header("Parâmetros do DBSCAN")
                eps = st.sidebar.slider("Epsilon (eps)", 0.1, 5.0, 1.5, 0.1)
                min_samples = st.sidebar.slider("Amostras Mínimas (min_samples)", 1, 15, 5, 1)

            if not features_to_use:
                st.warning("Por favor, selecione pelo menos uma feature para a análise.")
                return

            features_df = df[features_to_use]
            
            # --- LÓGICA DE ANÁLISE AGORA DELEGADA ---
            analyzer = ClusterAnalyzer(eps=eps, min_samples=min_samples)
            results = analyzer.analyze(features_df)
            # ----------------------------------------

            df['cluster'] = results.get("clusters")

            st.header("Resultados da Clusterização")
            col1, col2 = st.columns(2)
            col1.metric("Número de Clusters Encontrados", results.get("n_clusters"))
            col2.metric("Número de Anomalias (Ruído)", results.get("n_noise"), delta_color="inverse")

            principal_components = results.get("principal_components")
            if principal_components is None:
                st.warning("Selecione pelo menos 2 features para a visualização PCA.")
                return

            fig, ax = plt.subplots(figsize=(12, 8))
            for cluster_id in sorted(set(df['cluster'])):
                label = 'Anomalia (Ruído)' if cluster_id == -1 else f'Cluster {cluster_id}'
                color = 'red' if cluster_id == -1 else f'C{cluster_id}'
                marker = 'x' if cluster_id == -1 else 'o'
                indices = df[df['cluster'] == cluster_id].index
                ax.scatter(principal_components[indices, 0], principal_components[indices, 1],
                           label=label, c=color, marker=marker, s=100, alpha=0.7)

            ax.set_title("Visualização dos Clusters e Anomalias (via PCA)")
            ax.set_xlabel("Componente Principal 1")
            ax.set_ylabel("Componente Principal 2")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            st.write("### Tabela de Dados com Clusters:", df)

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()