from __future__ import annotations

import json
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional


class _DecimalEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        return super().default(o)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(i) for i in obj]
    return obj


class JsonlStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def append(self, name: str, record: Dict[str, Any]) -> None:
        path = self.data_dir / f"{name}.jsonl"
        entry = _to_jsonable(record)
        entry.setdefault("_ts", time.time())
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_lines(self, name: str, last_n: int = 0) -> List[Dict[str, Any]]:
        path = self.data_dir / f"{name}.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if last_n > 0:
            lines = lines[-last_n:]
        result = []
        for line in lines:
            if line.strip():
                result.append(json.loads(line))
        return result

    def read_json(self, name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        path = self.data_dir / f"{name}.json"
        if not path.exists():
            return default if default is not None else {}
        return json.loads(path.read_text(encoding="utf-8"))

    def write_json(self, name: str, data: Dict[str, Any]) -> None:
        path = self.data_dir / f"{name}.json"
        path.write_text(
            json.dumps(_to_jsonable(data), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def exists(self, name: str, ext: str = "json") -> bool:
        return (self.data_dir / f"{name}.{ext}").exists()
