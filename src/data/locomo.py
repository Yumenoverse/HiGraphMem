import json
from pathlib import Path
from typing import Any, Dict, List


def load_locomo(path: str) -> List[Dict[str, Any]]:
    root = Path(path)
    if root.is_file():
        files = [root]
    else:
        files = sorted(root.glob("*.json")) + sorted(root.glob("*.jsonl"))
        files += sorted((root / "data").glob("*.json")) + sorted((root / "data").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(
            "No LoCoMo json/jsonl files found. Login to HuggingFace and run scripts/download_data.sh."
        )
    rows: List[Dict[str, Any]] = []
    for file_path in files:
        if file_path.suffix == ".jsonl":
            with file_path.open("r", encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
        else:
            with file_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                rows.extend(payload)
            elif isinstance(payload, dict):
                rows.append(payload)
    return rows
