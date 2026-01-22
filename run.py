import sys
import runpy
from pathlib import Path

# Ensure the root folder is in sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# The script to run (relative to project root)
script = sys.argv[1]

# Turn 'differential_eqns/vanderpol.py' -> module path 'differential_eqns.vanderpol'
module_name = Path(script).with_suffix('').as_posix().replace('/', '.')

runpy.run_module(module_name, run_name="__main__")