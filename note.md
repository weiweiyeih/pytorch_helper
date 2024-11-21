```
import requests
from pathlib import Path

def download_file_from_github(raw_url: str, file_name: str, parent_folder: str=None):
  if parent_folder:
    data_path = Path(parent_folder)
    if not data_path.is_dir():
      data_path.mkdir(parents=True, exist_ok=True)

  HELPER_FILE = file_name
  RAW_GITHUB_URL = raw_url

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

```
from helper_cnn_classifier import CustomDataset
```
