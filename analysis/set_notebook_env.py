import os
import sys
from typing import Optional, Union
from pathlib import Path


# make sure project root is on the path when running from /analysis
project_root = os.path.abspath("..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import CapstoneDataLoader, plot_utils # this uses utils/__init__.py

def set_env(
    data_dir: Union[str, Path] = Path("../data"), 
    seed_value: Optional[int] = None, # get seed from .env if None
    min_ratings: Optional[int] = 0, 
    max_ratings: Optional[int] = None,
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
    return loader, plot_utils
    
if __name__ == "__main__":
    set_env(data_dir="../data")