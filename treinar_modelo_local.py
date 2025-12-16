# Copyright (c) 2025 Thauanny Kyssy Ramos Pereira. Todos os Direitos Reservados.
# ... (cabeçalho)

import pandas as pd
from src.analysis.cluster_analyzer import ClusterAnalyzer
from src.analysis.session_processor import SessionProcessor

# --- CONFIGURAÇÃO ---
LOCAL_DATASET_PATH = "gameplay_session.csv"
MODEL_PATH = "analyzer_model.joblib"
EPS_VALUE = 2.0

def main():
    print("--- INICIANDO TREINO OFFLINE COM DATASET LOCAL ---")
    try:
        df_session = pd.read_csv(LOCAL_DATASET_PATH)
    except FileNotFoundError:
        print(f"ERRO: Dataset '{LOCAL_DATASET_PATH}' não encontrado.")
        return

    print("A processar sessão de jogo e a extrair features...")
    processor = SessionProcessor()
    df_features = processor.process_session_df(df_session)

    if df_features.empty:
        print("ERRO: Nenhuma feature foi extraída.")
        return
        
    print(f"Foram extraídas features de {len(df_features)} janelas de análise.")

    num_features = df_features.shape[1]
    min_samples_calculado = 2 * num_features
    
    print(f"A treinar o modelo com eps={EPS_VALUE} e min_samples={min_samples_calculado}...")
    
    cluster_analyzer = ClusterAnalyzer(eps=EPS_VALUE, min_samples=min_samples_calculado)
    cluster_analyzer.fit(df_features)

    cluster_analyzer.save_model(MODEL_PATH)
    print(f"\n--- SUCESSO! Modelo pessoal treinado e salvo em '{MODEL_PATH}' ---")

if __name__ == "__main__":
    main()