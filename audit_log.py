"""Compact JSONL audit logging for every query_ai / query_ai_batch call.

AI_MCP_SERVER_HARDENING.md Priority 7: record enough metadata (ids, models,
template names, field lengths and hashes, prompt/response hashes, status,
duration) to reconstruct *what ran* without persisting resume content or API
keys.  Full prompt logging stays behind the existing opt-in
``save_debug_prompts`` mechanism.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("ai-mcp-server")

_REDACTED_KEYS = ("api_key", "authorization", "token", "secret")


class AuditLogger:
    """Append-only daily JSONL audit files with simple retention pruning."""

    def __init__(
        self,
        directory: str,
        enabled: bool = True,
        retention_days: int = 30,
    ) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        self.retention_days = retention_days
        self._pruned = False

    def _current_file(self) -> Path:
        return self.directory / f"audit-{datetime.now():%Y%m%d}.jsonl"

    def _prune_once(self) -> None:
        """Delete audit files older than the retention window (once per run)."""
        if self._pruned or self.retention_days <= 0:
            return
        self._pruned = True
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        try:
            for path in self.directory.glob("audit-*.jsonl"):
                try:
                    stamp = datetime.strptime(path.stem, "audit-%Y%m%d")
                except ValueError:
                    continue
                if stamp < cutoff:
                    path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Audit retention pruning failed: %s", exc)

    @staticmethod
    def _redact(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: ("[redacted]" if any(marker in key.lower() for marker in _REDACTED_KEYS)
                  else value)
            for key, value in record.items()
        }

    def record(self, record: Dict[str, Any]) -> Optional[Path]:
        """Append one audit record; never raises into the request path."""
        if not self.enabled:
            return None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._prune_once()
            payload = {"logged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            payload.update(self._redact(record))
            path = self._current_file()
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            return path
        except OSError as exc:
            logger.warning("Audit logging failed: %s", exc)
            return None
