"""Put the project root on sys.path so debug scripts can import play, config, etc.

Python adds a script's own folder to sys.path, not its parent. Every script in
this folder starts with `import _path` before any project import.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
