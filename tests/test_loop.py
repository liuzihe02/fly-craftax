import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.slow
@pytest.mark.parametrize("ablation,policy", [("full", "readout"), ("static", "random"),
                                            ("shuffled", "readout"), ("disconnected", "readout")])
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
    agent = build_agent(conn, drive, readout, BrainParams(),
                        wiring="shuffled" if ablation == "shuffled" else "full")
    env = make_env(2)
    logs = rollout(agent, env, jax.random.PRNGKey(0), n_actions=3, batch=2, ablation=ablation, policy=policy, keep_frames=True)
    assert logs["action"].shape == (3, 2) and logs["z"].shape == (3, 2, 6) and logs["sig"].shape == (3, 2, 6)
    assert logs["achievements"].shape == (3, 2, 22) and logs["frame"].shape == (3, 63, 63, 3)
    assert bool(jnp.isfinite(logs["z"]).all()) and bool((logs["active"] >= 0).all())
    assert float(logs["lamina"].mean()) > 1.0      # the biased lamina fires
    assert int(logs["action"].max()) <= 6


def test_summarise_survival_achievements_and_histogram():
    from flycraftax.loop import summarise
    t, b, n_ach = 4, 2, 22
    done = np.zeros((t, b), bool)
    done[2, 0] = True                       # env 0 dies on action index 2, env 1 never dies
    ach = np.zeros((t, b, n_ach), bool)
    ach[1:, 0, 3] = True                    # env 0 unlocked bit 3 one action before the fatal one
    ach[3, 0, 4] = True                     # bit 4 first shows after the fatal step: not credited
    ach[3, 1, 5] = True                     # env 1 unlocks bit 5 on the last logged step
    logs = dict(done=done, achievements=ach, action=np.zeros((t, b), np.int32),
                reward=np.full((t, b), 0.5, np.float32))
    r = summarise(logs, t)
    assert r["survival"] == [3, 4]
    assert r["achievements"][3] == 1 and r["achievements"][4] == 0 and r["achievements"][5] == 1
    assert sum(r["action_hist"]) == pytest.approx(1.0) and r["reward"] == pytest.approx(0.5)


@pytest.mark.slow
def test_linear_policy_rollout(conn):
    from flycraftax.brain import BrainParams
    from flycraftax.drive import build_drive
    from flycraftax.env import make_env
    from flycraftax.loop import build_agent, rollout
    from flycraftax.ppo import init_params
    from flycraftax.readout import Readout, readout_groups, std_floor
    from flycraftax.retina import build_retina
    drive = build_drive(conn, build_retina(conn), 100.0, lamina_mv=0.04)
    groups = readout_groups(conn)
    agent = build_agent(conn, drive, Readout(groups, np.zeros(6, np.float32), std_floor(groups)), BrainParams())
    assert agent.dn_idx.shape == (1314,)
    params = init_params(len(agent.dn_idx))
    env = make_env(2)
    logs = rollout(agent, env, jax.random.PRNGKey(0), 3, 2, policy="linear", params=params)
    assert logs["feats"].shape == (3, 2, 1314) and logs["logp"].shape == (3, 2) and logs["value"].shape == (3, 2)
    assert np.allclose(np.asarray(logs["logp"]), np.log(1 / 7), atol=1e-5)   # zero params: uniform policy
    assert logs["ep_return"].shape == (3, 2) and logs["ep_done"].dtype == bool


@pytest.mark.slow
def test_alternate_policy_is_forward_do(conn):
    from flycraftax.brain import BrainParams
    from flycraftax.drive import build_drive
    from flycraftax.env import DO, FORWARD, make_env
    from flycraftax.loop import build_agent, rollout
    from flycraftax.readout import Readout, readout_groups, std_floor
    from flycraftax.retina import build_retina
    drive = build_drive(conn, build_retina(conn), 100.0, lamina_mv=0.04)
    groups = readout_groups(conn)
    agent = build_agent(conn, drive, Readout(groups, np.zeros(6, np.float32), std_floor(groups)), BrainParams())
    env = make_env(2)
    logs = rollout(agent, env, jax.random.PRNGKey(0), 4, 2, policy="alternate")
    want = np.array([[FORWARD] * 2, [DO] * 2, [FORWARD] * 2, [DO] * 2])   # the open-loop control, per env
    assert np.array_equal(np.asarray(logs["action"]), want) and want[:, 0].tolist() == [1, 5, 1, 5]
    assert "feats" not in logs                          # keep_feats defaults off away from the linear policy
