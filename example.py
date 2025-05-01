from pathlib import Path
import os.path

CWD = Path(__file__).parent
print((CWD / "resources").exists())
