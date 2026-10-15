"""End-to-end change logic on the synthetic room with perfect geometry (no network models needed)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture(scope="module")
def scene():
    import eval_synthetic as ev

    from roomnesia import geometry

    return geometry.to_room_frame(ev.oracle_scene(n=16), 1.0)


def test_detects_exactly_the_changes(scene):
    from roomnesia import change

    a, b, clusters, info = change.detect(scene)
    kinds = sorted(c.kind for c in clusters)
    # chair moved (= one removal + one addition), bin removed, backpack added
    assert kinds == ["added", "added", "removed", "removed"]


def test_null_test_reports_nothing(scene):
    from roomnesia import change

    _, _, _, info = change.detect(scene)
    assert all(v["false_changes"] == 0 for v in info["null_test"].values())


def test_unobserved_is_never_changed(scene):
    """Surfaces the other recording never saw must be 'unverifiable', not 'removed'/'added'."""
    from roomnesia import change

    a, b, _, _ = change.detect(scene, null_test=False)
    for c in (a, b):
        unseen = (c.n_free + c.n_same) == 0
        assert np.all(c.state[unseen] == change.UNOBSERVED)


@pytest.fixture(scope="module")
def hard():
    import eval_synthetic as ev

    from roomnesia import geometry

    return geometry.to_room_frame(ev.oracle_scene(n=16, scenario="hard"), 1.0)


def _world_to_room(scene, n=16, scenario="hard"):
    """Similarity transform synthetic-world -> room frame, from the camera centres."""
    import eval_synthetic as ev

    from roomnesia.geometry import cam_centers

    Cw = np.concatenate([[p[0] for p in ev.gt_cameras(n, scenario)[v][0]] for v in "AB"])
    s, R, t = ev.umeyama(Cw, cam_centers(scene["extrinsic"]))
    return lambda p: s * R @ np.asarray(p) + t


def test_hard_abstains_on_unverifiable_and_finds_the_rug(hard):
    import make_synthetic as syn

    from roomnesia import appearance, change

    a, b, clusters, info = change.detect(hard, null_test=False)
    to_room = _world_to_room(hard)
    extras, abstain, rug = syn.hard_extras()
    reported = [np.median((a if c.kind == "removed" else b).xyz[c.idx], 0) for c in clusters]
    for g in abstain:  # never filmed / always hidden in the after-video: must NOT be reported
        p = to_room(g["center"])
        assert all(np.linalg.norm((r - p)[[0, 2]]) > 0.4 for r in reported), g["name"]
    excl = [a.xyz[c.idx] for c in clusters if c.kind == "removed"] + [b.xyz[c.idx] for c in clusters if c.kind == "added"]
    app, _ = appearance.appearance_changes(hard, a, info["valid"], info["tol_rel"], info["eps"], excl)
    p = to_room(rug.center)
    assert any(np.linalg.norm((np.median(a.xyz[c.idx], 0) - p)[[0, 2]]) < 0.3 for c in app), "rug not found"
