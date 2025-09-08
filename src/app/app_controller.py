import time
from typing import List, Dict, Any
from src.hardware.sensor_controller import SensorController
from src.domain.movement_test import MovementTest
from src.analysis.signal_analyzer import SignalAnalyzer

class AppController:
    """
    Controla o estado e a lógica de negócio da aplicação,
    independente da interface do utilizador.
    """

    def __init__(self):
        self.sensor_controller: SensorController | None = None
        self.analyzer = SignalAnalyzer()
        self.results: List[Dict[str, Any]] = []

    @property
    def is_connected(self) -> bool:
        """Verifica se o controlador está conectado."""
        return self.sensor_controller is not None

    def connect(self):
        """Tenta conectar-se ao controlador de sensores."""
        if not self.is_connected:
            self.sensor_controller = SensorController()

    def disconnect(self):
        """Desconecta-se do controlador de sensores."""
        if self.is_connected:
            self.sensor_controller.close()
            self.sensor_controller = None

    def run_test(self, test: MovementTest, progress_callback=None):
        """
        Executa um teste de movimento, analisa os resultados e guarda-os.
        Opcionalmente, usa um callback para reportar o progresso.
        """
        if not self.is_connected:
            raise RuntimeError("O controlador não está conectado para iniciar um teste.")

        timestamps, sensor_readings = [], []
        start_time = time.time()
        
        while time.time() - start_time < test.duration_seconds:
            try:
                data = self.sensor_controller.get_sensors_data()
                timestamps.append(time.time() - start_time)
                sensor_readings.append(data['accel_x'])
                
                if progress_callback:
                    progress = (time.time() - start_time) / test.duration_seconds
                    progress_callback(min(progress, 1.0))
                    
            except TimeoutError:
                continue
            
            time.sleep(0.01)

        if not sensor_readings:
            print("Aviso: Nenhum dado foi coletado durante o teste.")
            return

        sample_rate = len(timestamps) / test.duration_seconds
        fft_results = self.analyzer.find_tremor_frequency(sensor_readings, sample_rate)
        
        self.results.append({
            "name": test.name,
            "timestamps": timestamps,
            "readings": sensor_readings,
            "fft_results": fft_results,
            "sample_rate": sample_rate
        })