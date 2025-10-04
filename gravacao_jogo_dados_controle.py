# Copyright (c) 2025 Thauanny Kyssy Ramos Pereira. Todos os Direitos Reservados.
#
# Este software é propriedade confidencial e proprietária de Thauanny Kyssy Ramos Pereira.
# A utilização, cópia ou divulgação deste ficheiro só é permitida de acordo
# com os termos de um contrato de licença celebrado com o autor.

"""
Script para gravar dados de sensores e botões de um controle DualSense
durante uma sessão de jogo, salvando o resultado num ficheiro CSV.
"""

import time
import csv
from pydualsense import pydualsense

# --- CONFIGURAÇÕES ---
OUTPUT_FILENAME = "gameplay_session.csv"
LOGGING_FREQUENCY_HZ = 100

class GameDataLogger:
    def __init__(self):
        self.dualsense = None
        self.data_buffer = []
        self.latest_sensor_data = {
            'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0,
            'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0
        }

    def _setup_callbacks(self):
        """Configura os callbacks para atualizar os dados dos sensores."""
        self.dualsense.accelerometer_changed += self._on_accelerometer_update
        self.dualsense.gyro_changed += self._on_gyro_update

    def _on_accelerometer_update(self, x, y, z):
        """Atualiza os valores do acelerómetro."""
        self.latest_sensor_data['accel_x'] = x
        self.latest_sensor_data['accel_y'] = y
        self.latest_sensor_data['accel_z'] = z

    def _on_gyro_update(self, pitch, yaw, roll):
        """Atualiza os valores do giroscópio."""
        self.latest_sensor_data['gyro_x'] = pitch
        self.latest_sensor_data['gyro_y'] = yaw
        self.latest_sensor_data['gyro_z'] = roll

    def run(self):
        """Executa o fluxo principal de conexão e gravação."""
        try:
            self.dualsense = pydualsense()
            self.dualsense.init()
            print("Controlador DualSense encontrado e conectado.")
            self._setup_callbacks()
            
            input("\n>>> Pressione [Enter] para começar a gravar... <<<")
            print(f"\nGravação iniciada! A gravar dados a {LOGGING_FREQUENCY_HZ} Hz.")
            print("Jogue o seu jogo. Quando terminar, volte a este terminal e pressione [Ctrl+C] para parar.")

            self._logging_loop()

        except Exception as e:
            print(f"\nOcorreu um erro: {e}")
        finally:
            self.save_data()
            if self.dualsense:
                self.dualsense.close()
                print("Conexão com o controle fechada.")

    def _logging_loop(self):
        """Loop principal que grava os dados na frequência definida."""
        try:
            while True:
                timestamp = time.time()
                state = self.dualsense.state
                
                row = [
                    timestamp,
                    self.latest_sensor_data['accel_x'], self.latest_sensor_data['accel_y'], self.latest_sensor_data['accel_z'],
                    self.latest_sensor_data['gyro_x'], self.latest_sensor_data['gyro_y'], self.latest_sensor_data['gyro_z'],
                    int(state.R1), int(state.L1),
                    state.DpadUp, state.DpadDown, state.DpadLeft, state.DpadRight,
                    state.L2, state.R2
                ]
                self.data_buffer.append(row)
                time.sleep(1.0 / LOGGING_FREQUENCY_HZ)
        except KeyboardInterrupt:
            print("\nGravação interrompida pelo utilizador.")

    def save_data(self):
        """Salva os dados acumulados no buffer para um ficheiro CSV."""
        if not self.data_buffer:
            print("Nenhum dado para salvar.")
            return

        print(f"\nA salvar {len(self.data_buffer)} amostras de dados em '{OUTPUT_FILENAME}'...")
        header = [
            'timestamp', 
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'R1', 'L1',
            'DpadUp', 'DpadDown', 'DpadLeft', 'DpadRight',
            'L2_force', 'R2_force'
        ]
        
        with open(OUTPUT_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            # --- CORREÇÃO AQUI ---
            writer.writerows(self.data_buffer) # Estava 'data_guffer'
        
        print("Dados salvos com sucesso!")

if __name__ == "__main__":
    logger = GameDataLogger()
    logger.run()