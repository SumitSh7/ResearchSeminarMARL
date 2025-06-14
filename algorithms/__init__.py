# Package: algos
from .algo_iql import IQLAgent, train_iql
from .algo_mfq import MFQAgent, train_mfq

__all__ = [
    'IQLAgent',
    'MFQAgent',
    'train_iql',
    'train_mfq'
]