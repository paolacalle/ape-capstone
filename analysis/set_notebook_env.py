import os
import sys

# Try current dir as project root
project_root = os.getcwd()

if not os.path.isdir(os.path.join(project_root, "utils")):
    project_root = os.path.abspath(os.path.join(project_root, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Project root:", project_root)
print("Contents:", os.listdir(project_root))

from utils.seed_setter import SeedSetter

SeedSetter.set_seed()
