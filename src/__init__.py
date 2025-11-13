# src/__init__.py

import sys
from . import config as cfg
print("Initializing src package")
print("Python Path:", sys.path)
print(" Application model Selected:", cfg.file_name)   # Printing the file name selected
