import streamlit as st
from src.app.app_controller import AppController
from src.domain.movement_test import MovementTest
from src.utils.plotter import plot_test_results

class StreamlitApp:
    """
    Encapsula a lógica de renderização da interface gráfica.
    Delega a lógica de negócio para um AppController.
    """

    def __init__(self):
        st.set_page_config(page_title="Análise de Movimento", layout="wide")
        
        # Inicializa o controlador no estado da sessão
        if 'app_controller' not in st.session_state:
            st.session_state.app_controller = AppController()
        
        # Mantém uma referência ao controlador para facilitar o acesso
        self.controller: AppController = st.session_state.app_controller
        
        self.TESTS = {
            "Teste de Repouso": MovementTest(
                name="Teste de Repouso",
                instructions="Segure o controle parado na sua mão, apoiado na perna.",
                duration_seconds=10
            ),
            "Teste de Movimento Lento": MovementTest(
                name="Teste de Movimento Lento",
                instructions="Estique o braço e mova o controle LENTAMENTE da esquerda para a direita.",
                duration_seconds=15
            )
        }

    def _render_sidebar_controls(self):
        """Renderiza a barra lateral com os controlos principais."""
        with st.sidebar:
            st.header("Configuração")

            if not self.controller.is_connected:
                if st.button("🔌 Conectar ao Controle"):
                    with st.spinner("Procurando controlador..."):
                        try:
                            self.controller.connect()
                            st.success("Controlador conectado!")
                            st.rerun()
                        except ConnectionError as e:
                            st.error(f"Falha na conexão: {e}")
            else:
                st.success("✅ Controlador Conectado")
                if st.button("🔌 Desconectar"):
                    self.controller.disconnect()
                    st.rerun()

            selected_test_name = st.selectbox(
                "Escolha um teste para realizar:",
                options=list(self.TESTS.keys()),
                disabled=(not self.controller.is_connected)
            )
            selected_test = self.TESTS[selected_test_name]
            st.info(f"**Instruções:** {selected_test.instructions}\n\n**Duração:** {selected_test.duration_seconds}s.")

            if st.button("🚀 Iniciar Teste", type="primary", disabled=(not self.controller.is_connected)):
                placeholder = st.empty()
                
                # O callback permite que a lógica de negócio comunique o progresso à UI
                def progress_callback(progress):
                    placeholder.progress(progress, text=f"Progresso: {int(progress*100)}%")
                
                with st.spinner(f"Executando '{selected_test.name}'..."):
                    self.controller.run_test(selected_test, progress_callback)
                
                placeholder.empty()
                st.rerun()

    def _render_main_content(self):
        """Renderiza a área de conteúdo principal com os resultados."""
        st.header("Resultados")
        if not self.controller.results:
            st.info("Aguardando a execução de um teste.")
        else:
            last_result = self.controller.results[-1]
            st.subheader(f"Análise para: {last_result['name']}")
            
            _, _, dominant_freq, _ = last_result['fft_results']
            st.metric("Frequência de Pico Detectada (4-8 Hz)", f"{dominant_freq:.2f} Hz")
            
            fig = plot_test_results(
                time_axis=last_result['timestamps'], sensor_data=last_result['readings'],
                fft_results=last_result['fft_results'], test_name=last_result['name']
            )
            st.pyplot(fig)

    def run(self):
        """Executa o fluxo principal da aplicação Streamlit."""
        st.title("🔬 Análise de Movimento com Sensores")
        self._render_sidebar_controls()
        self._render_main_content()

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()