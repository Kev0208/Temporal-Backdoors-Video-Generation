"""
BadVideo Attack 
"""

from .config import AttackType, get_attack_config
from .pipeline import BadVideoPipeline

__all__ = [
    'AttackType',
    'get_attack_config',
    'BadVideoPipeline',
]

