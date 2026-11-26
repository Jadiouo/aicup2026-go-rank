class GogameError(Exception):
    pass


class DataFormatError(GogameError):
    pass


class SubmissionError(GogameError):
    pass


class ConfigError(GogameError):
    pass


from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class Issue:
    """A validation finding. Codes starting with 'W_' are warnings."""
    code: str
    row: int | None
    detail: str
