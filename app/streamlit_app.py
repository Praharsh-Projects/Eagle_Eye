from pathlib import Path
import sys

# Streamlit Cloud can execute this file with `app/` as cwd.
# Ensure repository root is importable so `src.*` modules resolve.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app.streamlit_app import main


if __name__ == "__main__":
    main()
