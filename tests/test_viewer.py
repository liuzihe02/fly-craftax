import numpy as np


def test_fly_sprites_shape_and_facing():
    from flycraftax.viewer import fly_sprites
    s = fly_sprites()
    assert s.shape == (5, 16, 16, 4) and s.dtype == np.uint8
    alpha = s[..., 3] > 0
    assert alpha.reshape(5, -1).sum(1).min() > 20              # every sprite has a body
    # facing sprites are not identical to each other, and left/right are mirror images
    assert not np.array_equal(s[0], s[2])
    assert np.array_equal(s[0], s[1][:, ::-1])


def test_compose_returns_image():
    from flycraftax.viewer import compose
    n = 50
    xy = np.random.default_rng(0).normal(size=(n, 2)).astype(np.float32)
    counts = np.zeros(n); counts[:5] = 3
    groups = {"forward": np.array([0, 1]), "backward": np.array([2]), "turn_l": np.array([3]), "turn_r": np.array([4]),
              "do": np.array([5]), "sleep": np.array([6, 7])}
    img = compose(np.full((576, 576, 3), 128.0), counts, xy, groups, np.array([0.2, -0.1, 1.5, -1.5, 0.0, 0.4]), 3,
                  dict(health=9, food=7, drink=5, energy=3), 1.0)
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8 and img.shape[0] > 400
