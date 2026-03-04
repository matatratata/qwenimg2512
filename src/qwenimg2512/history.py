"""History management for generations."""

import json
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class HistoryManager:
    def __init__(self, history_file: Path):
        self.history_file = history_file
        self.history: list[dict] = []
        self.load()

    def load(self) -> None:
        if self.history_file.exists():
            try:
                self.history = json.loads(self.history_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error("Failed to load history: %s", e)
                self.history = []
        else:
            self.history = []

    def save(self) -> None:
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            self.history_file.write_text(json.dumps(self.history, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.error("Failed to save history: %s", e)

    def add_entry(self, tab_name: str, output_path: str, params: dict) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "tab_name": tab_name,
            "output_path": output_path,
            "params": params
        }
        self.history.insert(0, entry)
        if len(self.history) > 200:
            self.history = self.history[:200]
        self.save()

    def get_history(self) -> list[dict]:
        return self.history
