import random
import numpy as np

class SeedSetter:
    @staticmethod
    def set_seed(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)