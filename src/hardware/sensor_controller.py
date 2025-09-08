import time
from typing import Dict, Optional
from pydualsense import pydualsense

class SensorController:
    """
    Gerencia a conexão e a leitura de dados de um controle
    Sony DualSense.
    """

    def __init__(self):
        self.dualsense = pydualsense()
        self._latest_sensor_data: Dict[str, float] = {}

        try:
            self.dualsense.init()
            print("Controlador DualSense encontrado e conectado.")

            self.dualsense.accelerometer_changed += self._on_accelerometer_update
            self.dualsense.gyro_changed += self._on_gyro_update
            time.sleep(0.5)
            if not self._latest_sensor_data:
                raise ConnectionError("Controlador conectado, mas não foi possível receber dados dos sensores.")

        except Exception as e:
            self.close()
            raise ConnectionError(f"Não foi possível encontrar ou inicializar um controle DualSense. Erro: {e}")

    def _on_accelerometer_update(self, x: float, y: float, z: float):
        """ Callback que é chamado quando os dados do acelerômetro mudam. """
        self._latest_sensor_data['accel_x'] = x
        self._latest_sensor_data['accel_y'] = y
        self._latest_sensor_data['accel_z'] = z

    def _on_gyro_update(self, pitch: float, yaw: float, roll: float):
        """ Callback que é chamado quando os dados do giroscópio mudam. """
        self._latest_sensor_data['gyro_x'] = pitch
        self._latest_sensor_data['gyro_y'] = yaw
        self._latest_sensor_data['gyro_z'] = roll

    def get_sensors_data(self) -> Dict[str, float]:
        """
        Retorna a leitura mais recente dos sensores armazenada pelos callbacks.
        """
        if not self._latest_sensor_data:
            raise TimeoutError("Dados dos sensores ainda não estão disponíveis.")
        return self._latest_sensor_data.copy()

    def close(self):
        """ Fecha a conexão com o controle. """
        if self.dualsense:
            self.dualsense.close()
            print("\nConexão com o controlador fechada.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()