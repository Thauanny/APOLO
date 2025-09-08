from dataclasses import dataclass

@dataclass
class MovementTest:
    """
    Representa um teste de movimento a ser realizado pelo usuário.
    """
    name: str
    instructions: str
    duration_seconds: int