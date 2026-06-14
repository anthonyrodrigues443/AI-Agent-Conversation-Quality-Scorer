"""Put the repo root on sys.path so `import src...` works under pytest from any cwd."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
