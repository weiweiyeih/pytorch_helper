```
import requests
from pathlib import Path

HELPER_FILE = "helper_cnn_classifier.py"
RAW_GITHUB_URL =

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path(HELPER_FILE).is_file():
  print(f"{HELPER_FILE} already exists, skipping download")
else:
  print(f"Downloading {HELPER_FILE}")
  # Note: you need the "raw" GitHub URL for this to work
  request = requests.get(RAW_GITHUB_URL)
  with open(HELPER_FILE, "wb") as f:
    f.write(request.content)
```
