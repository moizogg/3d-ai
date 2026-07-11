"""Progress heartbeat unit test."""

from __future__ import annotations

import time

from propertyscan.core.progress import ProgressHeartbeat


def test_heartbeat_emits_start_and_finish() -> None:
    lines: list[str] = []

    def capture(msg: str) -> None:
        lines.append(msg)

    with ProgressHeartbeat("unit_test", interval_s=60.0, print_fn=capture) as hb:
        hb.set_status("working")
        time.sleep(0.05)

    assert any("started" in x for x in lines)
    assert any("working" in x for x in lines)
    assert any("finished OK" in x for x in lines)
