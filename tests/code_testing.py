import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import train
from src.utils import generate_empty_dir

if __name__ == "__main__":
    generate_empty_dir.create_dir()