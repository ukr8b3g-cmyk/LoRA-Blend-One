from __future__ import annotations


class MergeError(ValueError):
    """A merge failure with source names for the caller's error display."""

    def __init__(self, code: str, sources: list[str], detail: str, *, source_paths: list[str] | None = None):
        self.code = code
        self.sources = list(sources)
        self.detail = detail
        self.source_paths = list(source_paths) if source_paths is not None else None
        super().__init__(detail)
