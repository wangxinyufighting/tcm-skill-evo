"""Make `import tcmskill` work without `pip install -e .` and change cwd to the repo root."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
