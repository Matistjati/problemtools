"""Compatibility shims for older Python versions."""

try:
    from enum import StrEnum  # Python 3.11+
except ImportError:
    import enum
    class StrEnum(str, enum.Enum):  # type: ignore[no-redef]
        """Backport of enum.StrEnum for Python < 3.11."""
        pass

try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing import TypeVar
    Self = TypeVar('Self')  # type: ignore[assignment,misc]
