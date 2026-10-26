"""Render PhysX rollout offscreen with the pinned Isaac Sim training visuals.

Live capture renders top every tick; policy-only capture renders three cameras
only at policy ticks. Physical colliders remain hidden from RGB.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

from training_render import training_jpeg, verify_camera_config

if TYPE_CHECKING:
    from render_initial_scene import RecordedVisualScene
    from pxr import Usd


_TRAINING_SRC = Path(__file__).resolve().parent / "runtime/training-render/src"
if str(_TRAINING_SRC) not in sys.path:
    sys.path.insert(0, str(_TRAINING_SRC))

from offline_rendering.isaacsim import (  # noqa: E402
    IsaacSimError,
    IsaacSimRenderer,
    _runtime_modules,
)
from raw_data import CAMERA_NAMES  # noqa: E402


class _ManualTimeline:
    """Keep the training renderer's stop calls from invalidating live PhysX."""

    def stop(self) -> None:
        # SimulationManager's timeline-stop callback destroys its simulation view.
        # The manual PhysX loop does not need a timeline stop for RGB capture.
        pass


class _PhysxRuntime:
    """Adapt the active headless Kit app and PhysX stage to the training renderer."""

    def __init__(self, stage: Any) -> None:
        import omni.kit.app
        import omni.usd

        self._stage = stage
        self._app = omni.kit.app.get_app()
        self._modules = replace(_runtime_modules(), timeline=_ManualTimeline())
        self._active_renderer: IsaacSimRenderer | None = None
        context = omni.usd.get_context()
        current = context.get_stage()
        if (self._app is None or current is None or stage is None
                or current.GetRootLayer().identifier != stage.GetRootLayer().identifier
                or current.GetSessionLayer().identifier != stage.GetSessionLayer().identifier):
            raise IsaacSimError("Renderer requires the active PhysX stage")
        self._stage_id = context.get_stage_id()

    def _stage_is_active(self) -> bool:
        import omni.usd

        context = omni.usd.get_context()
        return context.get_stage() is not None and context.get_stage_id() == self._stage_id

    def _require_open(self) -> tuple[Any, Any, Any]:
        if not self._stage_is_active():
            raise IsaacSimError("PhysX USD stage changed during headless rendering")
        return self._app, self._modules, self._stage


class HeadlessRenderer:
    """Capture training JPEG images, and optionally live top-camera RGB."""

    def __init__(self, scene: RecordedVisualScene, camera_config: dict, stage: Usd.Stage,
                 capture: str = "top_live") -> None:
        verify_camera_config(camera_config)
        if capture not in ("top_live", "policy_only"):
            raise ValueError(f"Unknown PhysX capture mode: {capture}")
        self._capture = capture
        from pxr import UsdGeom

        self._scene = scene
        self._runtime = _PhysxRuntime(stage)
        if (UsdGeom.GetStageMetersPerUnit(stage) != 1.0
                or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z):
            raise IsaacSimError("PhysX stage must use the training renderer's Z-up meters")
        self._closed = False
        self._renderer: IsaacSimRenderer | None = None
        self._wrist_updates = True
        self._hidden: list[tuple[Any, bool, Any]] = []
        try:
            # The imported Gprims provide PhysX collision; only recorded raw
            # visuals belong in the policy image. Never hide their ancestors.
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Gprim):
                    imageable = UsdGeom.Imageable(prim)
                    attr = imageable.GetVisibilityAttr()
                    if imageable.ComputeVisibility() != UsdGeom.Tokens.invisible:
                        self._hidden.append((attr, attr.HasAuthoredValueOpinion(), attr.Get()))
                        attr.Set(UsdGeom.Tokens.invisible)

            renderer = IsaacSimRenderer(
                runtime=self._runtime,
                scene=scene,
                width=camera_config["width"],
                height=camera_config["height"],
                znear=camera_config["near_m"],
                fovys=(camera_config["fovy_deg"],) * len(CAMERA_NAMES),
            )
            self._renderer = renderer
            self._runtime._active_renderer = renderer
        except BaseException as exc:
            try:
                self.close()
            except BaseException as cleanup_error:
                exc.add_note(f"Headless renderer cleanup also failed: {cleanup_error}")
            raise

    def render(
        self, state: np.ndarray, object_pose: np.ndarray, tick: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self._closed or self._renderer is None:
            raise IsaacSimError("Headless renderer is closed")
        if not isinstance(tick, int) or tick < 0:
            raise ValueError("tick must be a non-negative integer")
        if self._capture == "policy_only" and tick % 8:
            raise ValueError("Policy-only capture must not render between image ticks")
        self._runtime._require_open()
        anchor = tick % 8 == 0
        if self._capture == "top_live" and self._wrist_updates != anchor:
            for product in self._renderer._render_products[1:]:
                product.hydra_texture.set_updates_enabled(anchor)
            self._wrist_updates = anchor
        # PhysX owns world poses. Only /World/SpdRaw visuals and camera xforms
        # receive the current measured state; no recorded future rows are used.
        raw = self._renderer.render_rgb(self._scene.frame(state, object_pose))
        if anchor:
            policy = np.stack([training_jpeg(image)[1] for image in raw], axis=0)
            return (raw.copy() if self._capture == "top_live" else None), policy
        return raw[:1].copy(), None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None
                self._runtime._active_renderer = None
        finally:
            for attr, authored, previous in reversed(self._hidden):
                if authored:
                    attr.Set(previous)
                else:
                    attr.Clear()
            self._hidden.clear()
