"""Unit tests for Stage protocol and RunContext."""

from __future__ import annotations

from pathlib import Path

from propertyscan.core.config import load_config
from propertyscan.core.context import RunContext
from propertyscan.core.exceptions import StageError
from propertyscan.core.stage import Stage, StageResult


class _OkStage(Stage):
    name = "ok_stage"

    def execute(self, ctx: RunContext) -> StageResult:
        ctx.set("marker", 42)
        return StageResult(stage_name=self.name, success=True, metrics={"n": 1})


class _BoomStage(Stage):
    name = "boom_stage"

    def execute(self, ctx: RunContext) -> StageResult:
        raise RuntimeError("intentional failure")


def test_run_context_state_and_artifacts(tmp_path: Path) -> None:
    cfg = load_config(apply_env=False)
    ctx = RunContext(
        config=cfg,
        input_path=tmp_path / "in",
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        job_id="job-test",
    )
    assert ctx.work_dir.is_dir()
    art = ctx.artifact_dir("frames")
    assert art.is_dir()
    ctx.set("foo", "bar")
    assert ctx.require("foo") == "bar"


def test_stage_success_records_history(tmp_path: Path) -> None:
    cfg = load_config(apply_env=False)
    ctx = RunContext(
        config=cfg,
        input_path=tmp_path / "in",
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        job_id="job-ok",
    )
    result = _OkStage().run(ctx)
    assert result.success is True
    assert ctx.require("marker") == 42
    assert len(ctx.stage_history) == 1
    assert ctx.provenance is not None
    assert len(ctx.provenance.stages) == 1


def test_stage_failure_wraps_stage_error(tmp_path: Path) -> None:
    cfg = load_config(apply_env=False)
    ctx = RunContext(
        config=cfg,
        input_path=tmp_path / "in",
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        job_id="job-boom",
    )
    try:
        _BoomStage().run(ctx)
        assert False, "expected StageError"
    except StageError as err:
        assert err.stage_name == "boom_stage"
        assert "intentional failure" in str(err)


def test_write_metadata_and_provenance(tmp_path: Path) -> None:
    cfg = load_config(apply_env=False)
    ctx = RunContext(
        config=cfg,
        input_path=tmp_path / "in",
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        job_id="job-meta",
    )
    _OkStage().run(ctx)
    meta_path = ctx.write_metadata()
    prov_path = ctx.write_provenance()
    assert meta_path.is_file()
    assert prov_path is not None and prov_path.is_file()
