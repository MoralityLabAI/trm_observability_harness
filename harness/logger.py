from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
from .schemas import StepRecord

class JsonlLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: StepRecord) -> None:
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')

    def write_many(self, records: Iterable[StepRecord]) -> None:
        with self.path.open('a', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')
