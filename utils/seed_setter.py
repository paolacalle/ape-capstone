import os
import random
import numpy as np
from dotenv import load_dotenv

# hardcoded for now because IDK 
SEED = 18787288
class SeedSetter:
    @staticmethod
    def get_seed() -> int:
        print("="*40)
        print(f"Using seed: {SEED}")
        print("="*40)
        
        return SEED
    
    @staticmethod
    def set_seed(seed: int = None):
        if seed is None:
            seed = SEED
        random.seed(seed)
        np.random.seed(seed)
        return seed