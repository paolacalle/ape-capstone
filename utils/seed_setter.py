import os
import random
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class SeedSetter:
    @staticmethod
    def get_seed() -> int:
        seed = int(os.getenv("SEED"))
        
        print("="*40)
        print(f"Using seed: {seed}")
        print("="*40)
        
        return seed
    
    @staticmethod
    def set_seed(seed: int = None):
        if seed is None:
            seed = SeedSetter.get_seed()
        random.seed(seed)
        np.random.seed(seed)
        return seed