"""Per-source early success and bounded failure using the existing geometry scorers."""
from __future__ import annotations

from collections import deque
from pathlib import Path

from baseline_raw_rollout import _score


TASK_LIMIT_TICKS = {"hang_mug": 750, "pyramid": 1500, "toss_in_bin": 1800,
                    "unstack": 1200, "vowel_consonant_sort": 3150}
WINDOW_TICKS = 31


class TaskTermination:
    def __init__(self, scene: dict, assets: Path, limit_ticks: int):
        if type(limit_ticks) is not int or limit_ticks < WINDOW_TICKS - 1:
            raise ValueError("Task limit must allow one full second of 30 Hz observations")
        self.scene, self.assets, self.limit_ticks = scene, assets, limit_ticks
        self.history = deque(maxlen=WINDOW_TICKS)
        self.last_tick = -1
        self.terminal = None

    def finish_at_limit(self, tick: int, score: dict) -> dict:
        if tick != self.limit_ticks or score["status"] not in ("failure", "unsupported"):
            raise ValueError("Only unmet or unverifiable goals terminate at the task limit")
        status = score["status"]
        return {"status": status, "reason": score["reason"],
                "termination_reason": "timeout_goal_unmet" if status == "failure" else "goal_unverifiable",
                "score": score, "tick": tick, "inference_count": (tick + 7) // 8}

    def update(self, tick: int, object_pose, hand_positions) -> dict | None:
        if self.terminal is not None:
            return self.terminal
        if type(tick) is not int or tick != self.last_tick + 1 or tick > self.limit_ticks:
            raise ValueError("Source control ticks must advance consecutively within its task limit")
        self.history.append((object_pose, hand_positions))
        self.last_tick = tick
        if len(self.history) < WINDOW_TICKS:
            return None
        score = _score(self.scene, self.history, self.assets)
        if score["status"] == "success":
            self.terminal = {"status": "success", "reason": score["reason"],
                             "termination_reason": "goal_verified", "score": score,
                             "tick": tick, "inference_count": (tick + 7) // 8}
        elif tick == self.limit_ticks:
            self.terminal = self.finish_at_limit(tick, score)
        return self.terminal
