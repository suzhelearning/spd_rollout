"""A transient simulator fault must not erase evidence or become a final score."""
import json

import baseline_raw_rollout as rollout


def _row(episode):
    return {"split": "val", "task_id": "hang_mug", "episode_id": episode}


def test_retry_errors_selects_only_failed_or_unfinished_scenes(tmp_path):
    rows = [_row("failed"), _row("succeeded"), _row("unscored"), _row("interrupted")]
    for row, status in zip(rows, ("error", "success", "unsupported")):
        folder = tmp_path / row["split"] / row["task_id"] / row["episode_id"]
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps({**row, "status": status}))
    assert [row["episode_id"] for row in rollout._pending_rows(rows, tmp_path, retry_errors=False)] == ["interrupted"]
    assert [row["episode_id"] for row in rollout._pending_rows(rows, tmp_path, retry_errors=True)] == ["failed", "interrupted"]


def test_repeated_attempt_preserves_prior_infrastructure_failure(tmp_path):
    row = _row("failed")
    folder = tmp_path / row["split"] / row["task_id"] / row["episode_id"]
    folder.mkdir(parents=True)
    original = {**row, "status": "error", "reason": "PhysX disconnected", "isaac_log": "/tmp/first.log"}
    (folder / "result.json").write_text(json.dumps(original))
    attempt = folder / "attempt_2"
    attempt.mkdir()
    completed = {**row, "status": "failure", "reason": "not stable"}
    rollout._commit_result(folder, attempt, completed)
    assert json.loads((folder / "result.json").read_text()) == completed
    assert json.loads((attempt / "previous_error_result.json").read_text()) == original
    assert json.loads((attempt / "result.json").read_text()) == completed
