"""Funções utilitárias serão adicionadas conforme a pipeline evoluir."""


def validate_confidence(confidence: float) -> bool:
    """Retorna True se a confiança estiver no intervalo válido [0, 1]."""
    return 0.0 <= confidence <= 1.0
