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
        self.dualsense = None
        self._latest_sensor_data: Dict[str, float] = {}

        ds = pydualsense()
        ds.init()
        
        ds.accelerometer_changed += self._on_accelerometer_update
        ds.gyro_changed += self._on_gyro_update

        time.sleep(0.5)
        
        if not self._latest_sensor_data:
            try:
                ds.close()
            except:
                pass
            raise ConnectionError("Controlador conectado, mas não recebe dados dos sensores.")
        
        self.dualsense = ds

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
        if self.dualsense is not None:
            try:
                self.dualsense.close()
            except:
                pass
            self.dualsense = None
