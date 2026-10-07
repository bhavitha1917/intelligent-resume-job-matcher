"""Import-path bootstrap shared by every page. Streamlit runs each page as
an independent script, so this must be imported first by every page
(it's idempotent -- cheap to import repeatedly).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
