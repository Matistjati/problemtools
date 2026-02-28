import yaml
from pathlib import Path

from ._compat import StrEnum


class FormatVersion(StrEnum):
    LEGACY = 'legacy'
    V_2025_09 = '2025-09'

    @property
    def statement_directory(self) -> str:
        match self:
            case FormatVersion.LEGACY:
                return 'problem_statement'
            case FormatVersion.V_2025_09:
                return 'statement'

    @property
    def statement_extensions(self) -> list[str]:
        match self:
            case FormatVersion.LEGACY:
                return ['tex']
            case FormatVersion.V_2025_09:
                return ['md', 'tex']

    @property
    def output_validator_directory(self) -> str:
        match self:
            case FormatVersion.LEGACY:
                return 'output_validators'
            case FormatVersion.V_2025_09:
                return 'output_validator'

    # Support older version strings for backwards compatibility
    @classmethod
    def _missing_(cls, value):
        if value in ('2023-07', '2023-07-draft'):
            return cls.V_2025_09
        return None


def get_format_version(problem_root: Path) -> FormatVersion:
    """Loads the version from the problem in problem_root"""
    with open(problem_root / 'problem.yaml') as f:
        config: dict = yaml.safe_load(f) or {}
    return FormatVersion(config.get('problem_format_version', FormatVersion.LEGACY))
