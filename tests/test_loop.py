import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.slow
@pytest.mark.parametrize("ablation,policy", [("full", "readout"), ("static", "random")])
def test_rollout_shapes_and_finiteness(conn, ablation, policy):
    from flycraftax.brain import BrainParams
    from flycraftax.drive import build_drive
    from flycraftax.env import make_env
    from flycraftax.loop import build_agent, rollout
    from flycraftax.readout import Readout, readout_groups, std_floor
    from flycraftax.retina import build_retina
    drive = build_drive(conn, build_retina(conn), max_hz=100.0, lamina_mv=0.06)
    groups = readout_groups(conn)
    readout = Readout(groups, np.zeros(6, np.float32), std_floor(groups))
    agent = build_agent(conn, drive, readout, BrainParams(), ablation=ablation)
    env = make_env(2)
    logs = rollout(agent, env, jax.random.PRNGKey(0), n_actions=3, batch=2, ablation=ablation, policy=policy, keep_frames=True)
    assert logs["action"].shape == (3, 2) and logs["z"].shape == (3, 2, 6) and logs["sig"].shape == (3, 2, 6)
    assert logs["achievements"].shape == (3, 2, 22) and logs["frame"].shape == (3, 63, 63, 3)
    assert bool(jnp.isfinite(logs["z"]).all()) and bool((logs["active"] >= 0).all())
    assert float(logs["lamina"].mean()) > 1.0      # the biased lamina fires
    assert int(logs["action"].max()) <= 6
