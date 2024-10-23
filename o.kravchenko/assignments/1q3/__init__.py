import sys
from pathlib import Path

repo_dir = Path(__file__).parents[3]
sys.path.append(Path(repo_dir, "numpy/assignment1/cs231n"))
