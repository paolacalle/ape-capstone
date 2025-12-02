# APE Capstone Project
Assessing Professor Effectiveness (NYU DS-GA 1001)

# Overview
This project analyzes a scraped RateMyProfessor dataset as part of the DS-GA 1001 Capstone.
We answer 10 required questions (plus optional extra credit) involving:

- Gender bias

- Rating distributions

- Difficulty differences

- Tag-based analysis

- Regression models

- Classification models

- And broader insights from the qualitative data

All analysis is done using a structured, reproducible pipeline with a custom CapstoneDataLoader class and a consistent seed setting based on our N-number, as required in the assignment instructions.

# Project Structure
```graphql 
ape-capstone/
│
├── analysis/
│   ├── eda.ipynb              # EDA + initial exploration
│   ├── set_notebook_env.py    # Handles Python path + seeding for future notebooks
│   └── analysis.py            # Final .py script for the full analysis
│
├── utils/
│   ├── __init__.py
│   ├── seed_setter.py         # Seeds RNG using an N-number
│   └── data_loader.py         # CapstoneDataLoader (main pipeline class)
│
├── data/
│   ├── rmpCapstoneNum.csv
│   ├── rmpCapstoneTags.csv
│   └── rmpCapstoneQual.csv
│
├── plots/                     # All generated figures
│
└── report/
    └── capstone.pdf           # Final project report (generated later)

```

# Virtual Environment
```bash 
cd /path/to/project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Set .env file in utils folder
```bash 
SEED=18787288 # n-number 
```

# Key Components
### `SeedSetter` 
The assignment requires seeding the RNG using an N-number (losing points otherwise).
This class ensures that this seed is always set whenever we load our data.

### `CapstoneDataLoader` (the main pipeline)
This class handles:

| Step          | What It Does                                                              |
| ------------- | ------------------------------------------------------------------------- |
| **Load**      | Reads all three CSVs based on project structure                           |
| **Merge**     | Merges them column-wise (same row order guaranteed)                       |
| **Clean**     | Applies rating thresholds, handles missing data, removes gender conflicts |
| **Normalize** | Converts raw tag counts --> per-rating tag rates                          |
| **Prepare**   | Runs the entire pipeline in the correct order                             |

You typically only need:
```python 
from utils import CapstoneDataLoader

loader = CapstoneDataLoader(
    seed_value=18787288,  # required Net id
    min_ratings=5         # threshold for valid averages (there are filters so look into the file)
)

df = loader.prepare()     # final cleaned + tag-normalized dataset
```

The class also stores intermediate dataframes:
```python 
loader.num_df        # raw numeric data
loader.tags_df       # raw tag data
loader.qual_df       # department/university/state
loader.merged_df     # after concatenation
loader.cleaned_df    # after applying filtering rules
loader.prepared_df   # final dataset for modeling
```

This makes debugging, EDA, and transparency easy.

### Notebook Environment Setup (`set_notebook_env.py`)
Jupyter notebooks run from inside `/analysis/`, so Python can't automatically see project-level modules.

We fix that in one place:
```python
# analysis/set_notebook_env.py

import os, sys
project_root = os.path.abspath("..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

```

In your notebook:

```python
import set_notebook_env
loader = set_notebook_env.set_env(data_dir="../data/", min_ratings=0, max_ratings=15)
```

This:
- Adds the project root to sys.path
- Ensures imports like from utils import CapstoneDataLoader work
- Seeds the notebook environment

### Running the Full Pipeline in `analysis.py`

```python 
python analysis/analysis.py
```

This will:
- Load all data
- Clean / normalize
- Produce plots
- Begin answering the capstone questions