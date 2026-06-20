"""Tests for executor.main."""

from executor.main import main


def test_main_runs() -> None:
    """Smoke test: main() should not raise."""
    main()
