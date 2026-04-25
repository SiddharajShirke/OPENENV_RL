from __future__ import annotations

import os

from rl.train_ppo import _resolve_checkpoint_path


def test_resolve_checkpoint_path_handles_direct_file(monkeypatch) -> None:
    def _exists(path: str) -> bool:
        return path == "results/best_model/phase1_final.zip"

    monkeypatch.setattr(os.path, "exists", _exists)
    assert _resolve_checkpoint_path("results/best_model/phase1_final.zip") == "results/best_model/phase1_final.zip"


def test_resolve_checkpoint_path_adds_zip_suffix(monkeypatch) -> None:
    def _exists(path: str) -> bool:
        return path == "results/best_model/phase1_final.zip"

    monkeypatch.setattr(os.path, "exists", _exists)
    assert _resolve_checkpoint_path("results/best_model/phase1_final") == "results/best_model/phase1_final.zip"


def test_resolve_checkpoint_path_returns_none_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(os.path, "exists", lambda _path: False)
    assert _resolve_checkpoint_path("results/best_model/missing_model") is None
