"""Reading and checking one section of a TOML configuration file.

Every component's ``settings.py`` uses this module to turn its section of the
configuration file into trusted values. A section is read one key at a time;
each read checks the value's type and range and fills in the default when the
key is missing. After the last read, ``reject_unknown_keys`` reports any key
that was never read, which catches typos such as ``world_cuont``.

Example, for a made-up ``[example]`` section::

    section = SettingsSection(raw_section, "example")
    size = section.positive_integer("size", default=4)
    mode = section.choice("mode", ("fast", "slow"))   # required: no default
    section.reject_unknown_keys()

Any problem raises ``SettingsError`` with a message naming the section and key.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Marks a setting that has no default and must be written in the file.
REQUIRED = object()


class SettingsError(ValueError):
    """A configuration value is missing, of the wrong type, or out of range."""


class SettingsSection:
    """One table of a TOML file, such as ``[environment.simulation]``."""

    def __init__(self, values: dict[str, Any], name: str) -> None:
        self.values = values
        self.name = name
        self.read_keys: set[str] = set()

    # -- Single values --------------------------------------------------------

    def text(self, key: str, default: Any = REQUIRED) -> str:
        """A string, such as a run name."""
        if self._missing(key, default):
            return default
        value = self.values[key]
        if not isinstance(value, str) or not value:
            raise self._error(key, "must be non-empty text", value)
        return value

    def boolean(self, key: str, default: Any = REQUIRED) -> bool:
        """``true`` or ``false``, such as whether to write a report."""
        if self._missing(key, default):
            return default
        value = self.values[key]
        if not isinstance(value, bool):
            raise self._error(key, "must be true or false", value)
        return value

    def choice(self, key: str, options: Sequence[str], default: Any = REQUIRED) -> str:
        """One string out of a fixed set of options, such as ``"cpu"``."""
        if self._missing(key, default):
            return default
        value = self.values[key]
        if value not in options:
            allowed = ", ".join(repr(option) for option in options)
            raise self._error(key, f"must be one of {allowed}", value)
        return value

    def path(self, key: str, default: Any = REQUIRED) -> Path:
        """A file or folder path, relative to where the command is run.

        Whether the path exists is checked by whoever opens it.
        """
        if self._missing(key, default):
            return default
        value = self.values[key]
        if not isinstance(value, str) or not value:
            raise self._error(key, "must be a path written as text", value)
        return Path(value)

    def integer(
        self,
        key: str,
        default: Any = REQUIRED,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        """A whole number, optionally limited to a range (limits included)."""
        if self._missing(key, default):
            return default
        value = self.values[key]
        return self._check_integer(key, value, minimum, maximum)

    def positive_integer(self, key: str, default: Any = REQUIRED) -> int:
        """A whole number of at least 1, such as a count of worlds."""
        return self.integer(key, default, minimum=1)

    def number(
        self,
        key: str,
        default: Any = REQUIRED,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        """A real number, optionally limited to a range (limits included).

        Whole numbers are accepted and returned as floats, so ``reward = 1``
        and ``reward = 1.0`` mean the same.
        """
        if self._missing(key, default):
            return default
        value = self.values[key]
        return self._check_number(key, value, minimum, maximum)

    def positive_number(self, key: str, default: Any = REQUIRED) -> float:
        """A real number strictly greater than zero, such as a learning rate."""
        if self._missing(key, default):
            return default
        value = self._check_number(key, self.values[key], None, None)
        if value <= 0:
            raise self._error(key, "must be greater than zero", value)
        return value

    # -- Lists ----------------------------------------------------------------

    def number_range(
        self, key: str, default: Any = REQUIRED, minimum: float | None = None
    ) -> tuple[float, float]:
        """A ``[low, high]`` pair of numbers with ``low <= high``."""
        if self._missing(key, default):
            return default
        value = self.values[key]
        if not isinstance(value, list) or len(value) != 2:
            raise self._error(key, "must be a list of two numbers [low, high]", value)
        low = self._check_number(key, value[0], minimum, None)
        high = self._check_number(key, value[1], minimum, None)
        if low > high:
            raise self._error(key, "must have low <= high", value)
        return (low, high)

    def integer_list(
        self, key: str, default: Any = REQUIRED, minimum: int | None = None
    ) -> tuple[int, ...]:
        """A non-empty list of whole numbers, such as seeds or layer sizes."""
        if self._missing(key, default):
            return default
        value = self.values[key]
        if not isinstance(value, list) or not value:
            raise self._error(key, "must be a non-empty list of whole numbers", value)
        return tuple(self._check_integer(key, item, minimum, None) for item in value)

    def choice_list(
        self, key: str, options: Sequence[str], default: Any = REQUIRED
    ) -> tuple[str, ...]:
        """A list of options, each used at most once; may be empty."""
        if self._missing(key, default):
            return default
        value = self.values[key]
        if not isinstance(value, list):
            raise self._error(key, "must be a list", value)
        for item in value:
            if item not in options:
                allowed = ", ".join(repr(option) for option in options)
                raise self._error(key, f"may only contain {allowed}", value)
        if len(set(value)) != len(value):
            raise self._error(key, "must not repeat an option", value)
        return tuple(value)

    # -- Nested tables --------------------------------------------------------

    def table(self, key: str) -> dict[str, Any]:
        """A nested table, such as ``target`` inside ``[environment]``.

        Returns the table's raw values for its own settings class to read; a
        missing table is empty, so all of its defaults apply.
        """
        if self._missing(key, default={}):
            return {}
        value = self.values[key]
        if not isinstance(value, dict):
            raise self._error(key, "must be a table", value)
        return value

    # -- Finishing ------------------------------------------------------------

    def reject_unknown_keys(self) -> None:
        """Fail if the file contains a key that no read asked for."""
        unknown = sorted(set(self.values) - self.read_keys)
        if unknown:
            names = ", ".join(unknown)
            raise SettingsError(f"[{self.name}] has unknown settings: {names}")

    # -- Internal helpers -----------------------------------------------------

    def _missing(self, key: str, default: Any) -> bool:
        """Record that ``key`` was read; tell whether its default applies."""
        self.read_keys.add(key)
        if key in self.values:
            return False
        if default is REQUIRED:
            raise SettingsError(f"[{self.name}] {key} is required")
        return True

    def _check_integer(
        self, key: str, value: Any, minimum: int | None, maximum: int | None
    ) -> int:
        # TOML booleans are Python bools, which Python also counts as integers.
        if isinstance(value, bool) or not isinstance(value, int):
            raise self._error(key, "must be a whole number", value)
        self._check_limits(key, value, minimum, maximum)
        return value

    def _check_number(
        self, key: str, value: Any, minimum: float | None, maximum: float | None
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self._error(key, "must be a number", value)
        self._check_limits(key, value, minimum, maximum)
        return float(value)

    def _check_limits(
        self, key: str, value: float, minimum: float | None, maximum: float | None
    ) -> None:
        if minimum is not None and value < minimum:
            raise self._error(key, f"must be at least {minimum}", value)
        if maximum is not None and value > maximum:
            raise self._error(key, f"must be at most {maximum}", value)

    def _error(self, key: str, problem: str, value: Any) -> SettingsError:
        return SettingsError(f"[{self.name}] {key} {problem}, got {value!r}")
