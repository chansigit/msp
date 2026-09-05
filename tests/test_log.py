"""msp.log: one flushed stdout stream for msp and harness_bridge records."""

import io
import logging

import pytest

from msp import log as msp_log

_MARK = "_harness_bridge_handler"


@pytest.fixture(autouse=True)
def _drop_bridge_handlers():
    yield
    for name in ("msp", "harness_bridge"):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if getattr(handler, _MARK, False):
                logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)


def test_configure_routes_msp_and_bridge_records_with_bare_message_format():
    buf = io.StringIO()
    msp_log.configure(stream=buf)
    logging.getLogger("msp.integrate.pipeline").info("== PCA (10 comps on 30 HVGs)")
    logging.getLogger("harness_bridge.harness").info("== [inspect] agent: check_deg(5)")
    logging.getLogger("msp.integrate.pipeline").debug("hidden at INFO")
    assert buf.getvalue() == "== PCA (10 comps on 30 HVGs)\n== [inspect] agent: check_deg(5)\n"


def test_configure_replaces_its_own_handler_and_records_still_reach_caplog(caplog):
    first, second = io.StringIO(), io.StringIO()
    msp_log.configure(stream=first)
    msp_log.configure(stream=second)
    with caplog.at_level(logging.INFO, logger="msp"):
        logging.getLogger("msp.steps").info("once")
    assert first.getvalue() == ""
    assert second.getvalue() == "once\n"
    assert "once" in caplog.text


def test_ensure_attaches_default_handler_only_when_nothing_is_reachable(monkeypatch):
    # pytest keeps capture handlers on the root logger, which makes every
    # logger reachable; hide them so the test sees a bare interpreter.
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    msp_log.ensure()
    marked = [h for h in logging.getLogger("msp").handlers if getattr(h, _MARK, False)]
    assert len(marked) == 1
    msp_log.ensure()  # no-op: still one handler
    assert sum(getattr(h, _MARK, False) for h in logging.getLogger("msp").handlers) == 1

    buf = io.StringIO()
    msp_log.configure(stream=buf)  # explicit configuration replaces the default handler
    msp_log.ensure()  # and is left alone afterwards
    marked = [h for h in logging.getLogger("msp").handlers if getattr(h, _MARK, False)]
    assert len(marked) == 1 and marked[0].stream is buf
