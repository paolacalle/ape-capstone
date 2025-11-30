import os
import sys


# make sure project root is on the path when running from /analysis
project_root = os.path.abspath("..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import CapstoneDataLoader  # this uses utils/__init__.py

def set_env(
    data_dir: str, 
    seed_value: int = 18787288, 
    min_ratings: int = 5, 
    max_ratings: int = None,
    drop_missing_ratings: bool = True, 
    drop_inconsistent_gender: bool = True
) -> CapstoneDataLoader:
    """Set up the notebook environment, including random seeds for reproducibility."""
    
    loader = CapstoneDataLoader(
        data_dir=data_dir, 
        seed_value=seed_value, 
        min_ratings=min_ratings, 
        max_ratings=max_ratings,
        drop_missing_ratings=drop_missing_ratings, 
        drop_inconsistent_gender=drop_inconsistent_gender
    )
    return loader
    
if __name__ == "__main__":
    set_env(data_dir="../data")