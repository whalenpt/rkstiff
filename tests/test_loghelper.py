import logging
import pytest
from rkstiff.util import loghelper


def test_setup_logger_with_string_level(caplog):
    """Test that setup_logger creates a logger with correct level (string input)."""
    logger = loghelper.setup_logger("rkstiff.test_logger", "info")

    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1

    # Verify that a log message at INFO is captured
    with caplog.at_level(logging.INFO):
        logger.info("Hello world")

    assert "Hello world" in caplog.text


def test_setup_logger_with_int_level():
    """Test that setup_logger accepts integer loglevel."""
    logger = loghelper.setup_logger("rkstiff.test_int_logger", logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_setup_logger_invalid_string_level():
    """Test that setup_logger raises ValueError on invalid loglevel string."""
    with pytest.raises(ValueError, match="Invalid log level"):
        loghelper.setup_logger("rkstiff.bad_logger", "notalevel")


def test_setup_logger_does_not_duplicate_handlers():
    """Ensure setup_logger does not add duplicate handlers on repeated setup calls."""
    name = "rkstiff.no_duplicate"
    logger1 = loghelper.setup_logger(name, "INFO")
    logger2 = loghelper.setup_logger(name, "DEBUG")

    # Same logger object reused, no new handler added
    assert logger1 is logger2
    assert len(logger1.handlers) == 1


def test_set_log_level_string_and_int():
    """Test that set_log_level correctly updates level from string or int."""
    logger = logging.getLogger("rkstiff.level_test")

    # Start at WARNING
    loghelper.set_log_level(logger, "ERROR")
    assert logger.level == logging.ERROR

    # Switch to DEBUG using integer
    loghelper.set_log_level(logger, logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_set_log_level_invalid_string():
    """Test that set_log_level raises ValueError for invalid log level."""
    logger = logging.getLogger("rkstiff.invalid_level")
    with pytest.raises(ValueError, match="Invalid log level"):
        loghelper.set_log_level(logger, "WRONGLEVEL")


def test_get_solver_logger_uses_class_name(monkeypatch):
    """Test that get_solver_logger uses solver class name in logger name."""

    class DummySolver:
        __name__ = "DummySolver"

    setup_spy = []

    def fake_setup_logger(name, level):
        setup_spy.append((name, level))
        return logging.getLogger(name)

    monkeypatch.setattr(loghelper, "setup_logger", fake_setup_logger)

    logger = loghelper.get_solver_logger(DummySolver, "DEBUG")
    assert isinstance(logger, logging.Logger)
    assert setup_spy == [("rkstiff.DummySolver", "DEBUG")]


def test_get_solver_logger_creates_configured_logger():
    """Integration test for get_solver_logger actual logger behavior."""

    class DummySolver:
        pass

    logger = loghelper.get_solver_logger(DummySolver, "INFO")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "rkstiff.DummySolver"
    assert logger.level == logging.INFO
