import sys
import runpy
from pathlib import Path

# Ensure the root folder is in sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# The script to run (relative to project root)
script = sys.argv[1]

runpy.run_path(script)
