"""
Ponto de entrada principal para a aplicação de Análise de Movimento.

Este script utiliza o subprocess para lançar a interface do utilizador
construída com Streamlit.
"""
import sys
import subprocess
from pathlib import Path

def main() -> None:
    """
    Encontra o caminho para o script da UI do Streamlit e o executa
    usando o comando 'streamlit run'.
    """
    project_root = Path(__file__).parent
    streamlit_script_path = project_root / "src" / "app" / "streamlit_ui.py"

    print(f"Lançando a aplicação Streamlit a partir de: {streamlit_script_path}")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(streamlit_script_path)
    ]

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print("\nErro: O comando 'streamlit' não foi encontrado.")
        print("Certifique-se de que o ambiente virtual está ativado e o streamlit está instalado.")
    except subprocess.CalledProcessError as e:
        print(f"\nA aplicação Streamlit terminou com um erro: {e}")

if __name__ == "__main__":
    main()