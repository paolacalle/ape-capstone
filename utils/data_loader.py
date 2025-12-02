
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import pandas as pd
from .seed_setter import SeedSetter as ss

class CapstoneDataLoader:
    """
    Object-oriented helper for the APE capstone dataset.

    Typical usage:
        loader = CapstoneDataLoader(seed_value=18787288, min_ratings=5)
        df = loader.prepare()

        # access intermediate stages
        loader.num_df, loader.tags_df, loader.qual_df
        loader.merged_df
        loader.cleaned_df
        loader.cleaning_info
    """
    
    def __init__(
        self, 
        data_dir : Optional[str |Path] = Path("./data"),
        seed_value : int = None, # get seed from .env if None
        min_ratings : int = 5,
        max_ratings : int = None,
        drop_missing_ratings : bool = True,
        drop_inconsistent_gender : bool = True
    ) -> None:  
        
        # find the csv files
        if data_dir is None:
            self.data_dir = Path(__file__).resolve().parent / "data"
        else:
            self.data_dir = Path(data_dir)
            
        self.seed_value = seed_value
        self.min_ratings = min_ratings
        self.max_ratings = max_ratings
        self.drop_missing_ratings = drop_missing_ratings
        self.drop_inconsistent_gender = drop_inconsistent_gender
        
        # placeholders for dataframes
        self.num_df: Optional[pd.DataFrame] = None
        self.tags_df: Optional[pd.DataFrame] = None
        self.qual_df: Optional[pd.DataFrame] = None
        self.merged_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.prepared_df: Optional[pd.DataFrame] = None
        
        self.cleaning_info: Dict[str, int] = {}
        
    @property
    def num_cols(self) -> List[str]:
        NUM_COLS: List[str] = [
            "avg_rating",            # 1 
            "avg_difficulty",        # 2
            "num_ratings",           # 3
            "pepper",                # 4
            "would_take_again_prop", # 5
            "num_online_ratings",    # 6
            "male",                  # 7
            "female",                # 8
        ]
        return NUM_COLS
    @property
    def tag_cols(self) -> List[str]:
        TAG_COLS: List[str] = [
            "tough_grader",
            "good_feedback",
            "respected",
            "lots_to_read",
            "participation_matters",
            "no_skip",          # “Don’t skip class or you will not pass”
            "lots_of_hw",
            "inspirational",
            "pop_quizzes",
            "accessible",
            "papers",           # “So many papers”
            "clear_grading",
            "hilarious",
            "test_heavy",
            "few_things",       # “Graded by few things”
            "amazing_lectures",
            "caring",
            "extra_credit",
            "group_projects",
            "lecture_heavy",
        ]
        return TAG_COLS

    @property
    def qual_cols(self) -> List[str]:
        QUAL_COLS: List[str] = ["major", "university", "state"]
        return QUAL_COLS
        
    # --- Main preparation method ---
    def set_seed(self) -> None:
        """Set the random seed for reproducibility."""
        ss.set_seed(self.seed_value)
        
    def load_raw(
        self, 
        num_filename: str = "rmpCapstoneNum.csv",
        tags_filename: str = "rmpCapstoneTags.csv",
        qual_filename: str = "rmpCapstoneQual.csv"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the raw CSV files into dataframes."""
        num_path = self.data_dir / num_filename
        tags_path = self.data_dir / tags_filename
        qual_path = self.data_dir / qual_filename
        
        self.num_df = pd.read_csv(num_path)
        self.tags_df = pd.read_csv(tags_path)
        self.qual_df = pd.read_csv(qual_path)
        
        # quick sanity check
        assert len(self.num_df) == len(self.tags_df) == len(self.qual_df), \
            "Data files have inconsistent number of rows."
        
        return self.num_df, self.tags_df, self.qual_df
    
    def merge(self) -> pd.DataFrame:
        """Merge the numerical, tag, and qualitative dataframes."""
        if self.num_df is None or self.tags_df is None or self.qual_df is None:
            raise RuntimeError("Dataframes not loaded. Call load_raw() first.")
        
        # combine dataframes side by side (by columns)
        merged = pd.concat([self.num_df, self.tags_df, self.qual_df], axis=1)
        
        # quick sanity check
        expected_cols = len(self.num_cols) + len(self.tag_cols) + len(self.qual_cols)
        assert merged.shape[1] == expected_cols, \
            "Merged dataframe has unexpected number of columns."
            
        merged.columns = self.num_cols + self.tag_cols + self.qual_cols
        self.merged_df = merged
        
        return self.merged_df
    
    def clean(self) -> pd.DataFrame:
        """Clean the merged dataframe according to specified criteria.
        
        Rules: 
        - Drop low-rating-count professors (num_ratings < min_ratings).
        - Drop high-rating-count professors (num_ratings > max_ratings).
        - Drop rows with missing avg_rating / avg_difficulty (optional).
        - Drop rows where both male and female == 1 (optional).
        """
        if self.merged_df is None:
            raise RuntimeError("Merged dataframe not available. Call merge() first.")
        
        df = self.merged_df.copy()
        info : Dict[str, int] = {}
        
        # 1 -- Drop low-rating-count professors
        if self.min_ratings is not None:
            before = len(df)
            df = df[df["num_ratings"] >= self.min_ratings]
            dropped_count = before - len(df)
            info["dropped_low_rating_count"] = dropped_count
            
        if self.max_ratings is not None:
            before = len(df)
            df = df[df["num_ratings"] <= self.max_ratings]
            dropped_count = before - len(df)
            info["dropped_high_rating_count"] = dropped_count
            
        # 2 -- Drop rows with missing avg_rating / avg_difficulty
        if self.drop_missing_ratings:
            before = len(df)
            df = df.dropna(subset=["avg_rating", "avg_difficulty"])
            dropped_count = before - len(df)
            info["dropped_missing_ratings"] = dropped_count
            
        # 3 -- Drop rows where both male and female == 1
        if self.drop_inconsistent_gender:
            before = len(df)
            df = df[~((df["male"] == 1) & (df["female"] == 1))]
            dropped_count = before - len(df)
            info["dropped_inconsistent_gender"] = dropped_count
            
            
        # reset the index, so it does not reflect the dropped rows
        df = df.reset_index(drop=True)
        
        self.cleaning_info = info
        self.cleaned_df = df
        
        return self.cleaned_df
    
    def add_professor_relative_tag_rates(
        self, 
        prefix: str = "tag_professor_relative_"
    ) -> pd.DataFrame:
        
        df = self.cleaned_df.copy()
        
        # all the tag columns
        raw_tags = df[self.tag_cols]
        
        # compute the sum for each professor
        # this is the weight of the professor's total tags
        row_sums = raw_tags.sum(axis=1)
        
        # avoid division-by-zero by replacing zeros with NA
        row_sums = row_sums.replace(0, pd.NA)
        
        # calculate relative tag rates
        # this is calculated by dividing each tag count 
        # by the total tag count for that professor
        # weighs each tag equally
        rel_tags = raw_tags.div(row_sums, axis=0).fillna(0.0)
        
        # add the relative tag rates to the dataframe with the specified prefix
        for col in rel_tags.columns:
            df[f"{prefix}{col}"] = rel_tags[col]
            
        # sanity check: ensure no NaN values in the new columns
        assert not df[[f"{prefix}{col}" for col in rel_tags.columns]].isna().any().any(), \
            "NaN values found in relative tag rate columns."
            
            
        self.prepared_df = df
    
        return self.prepared_df
    
    def prepare(self) -> pd.DataFrame:
        """
        Run the full pipeline:
        - set_seed
        - load_raw
        - merge
        - clean
        - add_tag_rates

        Returns the final prepared dataframe.
        """
        self.set_seed()
        self.load_raw()
        self.merge()
        self.clean()
        
        # add any other logic as needed 
        
        return self.prepared_df
    
    