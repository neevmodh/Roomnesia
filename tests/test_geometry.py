import numpy as np

from roomnesia.geometry import backproject, project
from roomnesia.reconstruct import _unrotate


def _cam(yaw=0.3, t=(0.2, -0.1, 1.5)):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.concatenate([R, np.array(t)[:, None]], 1), np.array([[300.0, 0, 160], [0, 300, 120], [0, 0, 1]])


def test_backproject_project_roundtrip():
    E, K = _cam()
    D = np.random.default_rng(0).uniform(1, 4, (240, 320))
    P, (v, u) = backproject(D, K, E)
    uv, z = project(P.reshape(-1, 3), K, E)
    assert np.allclose(z, D.ravel(), atol=1e-6)
    assert np.allclose(uv[:, 0], u.ravel(), atol=1e-6) and np.allclose(uv[:, 1], v.ravel(), atol=1e-6)


def test_portrait_unrotate_is_exact():
    """Running VGGT on rotated portrait frames and rotating back must give the same 3D points."""
    E, K = _cam()
    H, W = 240, 160
    K[0, 2], K[1, 2] = W / 2, H / 2
    D = np.random.default_rng(1).uniform(1, 3, (1, H, W))
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    Dr = np.rot90(D, k=-1, axes=(1, 2))
    Er = (Rz @ E)[None]
    Kr = np.array([[[K[1, 1], 0, H - K[1, 2]], [0, K[0, 0], K[0, 2]], [0, 0, 1]]])
    D2, _, E2, K2 = _unrotate(Dr, Dr, Er, Kr, H)
    P0, _ = backproject(D[0], K, E)
    P2, _ = backproject(D2[0], K2[0], E2[0])
    assert np.abs(P0 - P2).max() < 1e-9
