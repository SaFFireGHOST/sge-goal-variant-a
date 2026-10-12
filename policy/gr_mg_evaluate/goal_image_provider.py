# Copyright (2024) Bytedance Ltd. and/or its affiliates
# Additions (c) 2026 — local extensions for user-supplied goal-image evaluation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Helpers for sourcing the *final* goal image used to condition GR-MG at
evaluation time.

Two providers are supported:

* ``IP2PGoalImageProvider`` — the original GR-MG behaviour: a goal image is
  generated on-the-fly by the InstructPix2Pix model every K env steps,
  conditioned on the language instruction and current observation. This
  effectively re-plans intermediate sub-goals.

* ``ManifestGoalImageProvider`` — the **user-supplied** mode required for our
  experiment. The user passes a JSON manifest mapping each test sample
  ``(sequence_idx, subtask_idx)`` to a single PNG/JPG file on disk. That image
  is treated as the *final* goal image for the entire subtask rollout, i.e. the
  same goal image is returned for every env step of that subtask. No subgoal
  re-planning happens.

Both providers expose the same callable signature::

    goal_rgb = provider.get_goal(sequence_idx, subtask_idx, subtask_name,
                                 step_idx, lang_annotation, progress, obs)

returning a single ``np.ndarray`` (H, W, 3) ``uint8`` image.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image


def _load_image_as_static_rgb(path: str) -> np.ndarray:
    """Load an RGB image and resize to the CALVIN static-cam shape (200, 200, 3)."""
    img = Image.open(path).convert("RGB")
    if img.size != (200, 200):
        img = img.resize((200, 200), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


class IP2PGoalImageProvider:
    """Wraps the original GR-MG IP2P-based goal generator.

    Re-plans a goal image every ``replan_every`` env steps using the language
    instruction and the latest static-cam observation, exactly as in the stock
    GR-MG eval loop.
    """

    def __init__(self, ip2p_model, replan_every: int = 20):
        self.ip2p_model = ip2p_model
        self.replan_every = replan_every
        self._cached_goal: Optional[np.ndarray] = None
        self._last_subtask_key: Optional[str] = None

    def reset(self) -> None:
        self._cached_goal = None
        self._last_subtask_key = None

    def get_goal(
        self,
        sequence_idx: int,
        subtask_idx: int,
        subtask_name: str,
        step_idx: int,
        lang_annotation: str,
        progress: int,
        obs: Dict[str, Any],
    ) -> np.ndarray:
        key = f"{sequence_idx}_{subtask_idx}"
        if key != self._last_subtask_key:
            # New subtask — clear cache.
            self._cached_goal = None
            self._last_subtask_key = key

        if step_idx % self.replan_every == 0 or self._cached_goal is None:
            static_rgb = obs["rgb_obs"]["rgb_static"]
            text = (
                lang_annotation
                + f".And {progress}% of the instruction has been finished."
            )
            self._cached_goal = self.ip2p_model.inference([static_rgb], [text])[0]
        return self._cached_goal

    @property
    def replans(self) -> bool:
        return True


class ManifestGoalImageProvider:
    """User-supplied goal images, one per ``(sequence_idx, subtask_idx)`` pair.

    The manifest is a JSON file with one of the following shapes:

    1. ``{"0_0": "path/to/goal_0_0.png", "0_1": "...", ...}``
       Keys are ``"<sequence_idx>_<subtask_idx>"`` strings.

    2. ``{"images": ["path/to/goal_0_0.png", "path/to/goal_0_1.png", ...]}``
       A flat list. Indexing follows the order in which the evaluation visits
       test samples, i.e. for each sequence ``s`` in ``[0, NUM_SEQUENCES)`` and
       each subtask ``t`` in ``[0, len(seq))``, samples are emitted in
       ``(s, t)`` order. We flatten using
       ``flat_idx = sequence_idx * subtasks_per_sequence + subtask_idx``.
       ``subtasks_per_sequence`` defaults to 5 (the CALVIN long-horizon chain
       length) but can be overridden via the ``"subtasks_per_sequence"`` key
       at the top level of the JSON.

    Paths inside the manifest may be absolute or relative to ``manifest_path``'s
    parent directory.

    The same goal image is returned for *every* environment step of the
    corresponding subtask — this enforces the "final-goal-image only, no
    subgoals" constraint.
    """

    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Goal-image manifest not found: {self.manifest_path}"
            )
        with open(self.manifest_path, "r") as f:
            data = json.load(f)

        self._root = self.manifest_path.parent
        self._dict_mode: bool = False
        self._list: list = []
        self._dict: Dict[str, str] = {}
        self._subtasks_per_sequence: int = int(data.get("subtasks_per_sequence", 5))

        if isinstance(data, dict) and "images" in data:
            self._list = list(data["images"])
        elif isinstance(data, list):
            self._list = list(data)
        elif isinstance(data, dict):
            self._dict_mode = True
            self._dict = {
                k: v
                for k, v in data.items()
                if k not in {"subtasks_per_sequence"}
            }
        else:
            raise ValueError(
                f"Unsupported manifest format in {self.manifest_path}: "
                f"expected dict or list, got {type(data).__name__}"
            )

        self._cache: Dict[str, np.ndarray] = {}
        self._last_subtask_key: Optional[str] = None

    def reset(self) -> None:
        self._last_subtask_key = None

    def _resolve(self, raw_path: str) -> Path:
        p = Path(raw_path).expanduser()
        if not p.is_absolute():
            p = (self._root / p).resolve()
        return p

    def _lookup_path(self, sequence_idx: int, subtask_idx: int) -> Path:
        key = f"{sequence_idx}_{subtask_idx}"
        if self._dict_mode:
            if key not in self._dict:
                raise KeyError(
                    f"Manifest {self.manifest_path} is missing an entry for "
                    f"test sample '{key}'. Add a key '<sequence_idx>_<subtask_idx>'."
                )
            return self._resolve(self._dict[key])
        flat_idx = sequence_idx * self._subtasks_per_sequence + subtask_idx
        if flat_idx >= len(self._list):
            raise IndexError(
                f"Manifest {self.manifest_path} has only {len(self._list)} entries "
                f"but evaluation reached flat index {flat_idx} "
                f"(sequence_idx={sequence_idx}, subtask_idx={subtask_idx}, "
                f"subtasks_per_sequence={self._subtasks_per_sequence})."
            )
        return self._resolve(self._list[flat_idx])

    def get_goal(
        self,
        sequence_idx: int,
        subtask_idx: int,
        subtask_name: str,
        step_idx: int,
        lang_annotation: str,
        progress: int,
        obs: Dict[str, Any],
    ) -> np.ndarray:
        key = f"{sequence_idx}_{subtask_idx}"
        if key != self._last_subtask_key:
            self._last_subtask_key = key
        if key not in self._cache:
            path = self._lookup_path(sequence_idx, subtask_idx)
            if not path.exists():
                raise FileNotFoundError(
                    f"Goal image for test sample '{key}' not found at {path}"
                )
            self._cache[key] = _load_image_as_static_rgb(str(path))
        return self._cache[key]

    @property
    def replans(self) -> bool:
        return False


class StaticGoalImageProvider:
    """Returns one fixed goal image for the entire run. Useful for the
    visualisation script when evaluating a single chosen test sample.
    """

    def __init__(self, image_path: str):
        self._goal = _load_image_as_static_rgb(image_path)

    def reset(self) -> None:  # no state
        pass

    def get_goal(self, *args, **kwargs) -> np.ndarray:
        return self._goal

    @property
    def replans(self) -> bool:
        return False
