# src/hardware/sensor_controller.py

import time
from typing import Dict, List
from pydualsense import pydualsense

class SensorController:
    """
    Gerencia a conexão e a leitura de dados de um controle
    Sony DualSense, incluindo sensores de movimento e botões.
    """
    def __init__(self):
        self.dualsense = pydualsense()
        self._latest_sensor_data: Dict[str, float] = {}
        self.button_press_timestamps: List[float] = [] # Lista para guardar os cliques

        try:
            self.dualsense.init()
            print("Controlador DualSense encontrado e conectado.")

            # Callbacks existentes para os sensores de movimento
            self.dualsense.accelerometer_changed += self._on_accelerometer_update
            self.dualsense.gyro_changed += self._on_gyro_update

            # --- NOVIDADE: Callback para o botão 'cross' (X) ---
            self.dualsense.cross_pressed += self._on_cross_button_press

            time.sleep(0.5)
            if not self._latest_sensor_data:
                raise ConnectionError("Controlador conectado, mas não foi possível receber dados dos sensores.")
        except Exception as e:
            self.close()
            raise ConnectionError(f"Não foi possível encontrar ou inicializar um controle DualSense. Erro: {e}")

    # --- NOVIDADE: Função que é chamada a cada clique no botão X ---
    def _on_cross_button_press(self, state: bool):
        """Callback que é chamado quando o estado do botão X muda."""
        if state: # Apenas regista quando o botão é pressionado (não quando é solto)
            self.button_press_timestamps.append(time.time())

    # --- NOVIDADE: Métodos para gerir os timestamps dos cliques ---
    def start_tapping_test(self):
        """Limpa a lista de timestamps para iniciar um novo teste de tapping."""
        self.button_press_timestamps = []

    def get_tapping_results(self) -> List[float]:
        """Retorna a lista de timestamps dos cliques desde o início do teste."""
        return self.button_press_timestamps

    def _on_accelerometer_update(self, x: float, y: float, z: float):
        self._latest_sensor_data['accel_x'] = x
        self._latest_sensor_data['accel_y'] = y
        self._latest_sensor_data['accel_z'] = z

    def _on_gyro_update(self, pitch: float, yaw: float, roll: float):
        self._latest_sensor_data['gyro_x'] = pitch
        self._latest_sensor_data['gyro_y'] = yaw
        self._latest_sensor_data['gyro_z'] = roll

    def get_sensors_data(self) -> Dict[str, float]:
        if not self._latest_sensor_data:
            raise TimeoutError("Dados dos sensores ainda não estão disponíveis.")
        return self._latest_sensor_data.copy()

    def close(self):
        if self.dualsense:
            self.dualsense.close()
            print("\nConexão com o controlador fechada.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()