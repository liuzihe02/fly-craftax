import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_gae_matches_reference():
    from flycraftax.ppo import gae
    r = jnp.array([[1.0], [0.0], [1.0]]); v = jnp.array([[0.5], [0.5], [0.5]]); d = jnp.array([[False], [False], [True]])
    adv, tgt = gae(r, v, d, jnp.array([0.5]), 0.9, 1.0)
    # backwards: t=2 done: delta = 1 - 0.5 = 0.5; t=1: 0 + 0.9*0.5 - 0.5 + 0.9*0.5 = 0.4; t=0: 1 + 0.45 - 0.5 + 0.9*0.4 = 1.31
    assert np.allclose(np.asarray(adv)[:, 0], [1.31, 0.4, 0.5], atol=1e-6)
    assert np.allclose(np.asarray(tgt), np.asarray(adv) + np.asarray(v))


def test_loss_is_finite_and_uniform_policy_has_full_entropy():
    from flycraftax.ppo import init_params, loss, policy
    p = init_params(5)
    feats = jnp.ones((4, 5))
    logits, value = policy(p, feats)
    assert logits.shape == (4, 7) and value.shape == (4,)
    batch = dict(feats=feats, action=jnp.array([0, 1, 2, 3]), logp_old=jnp.full(4, np.log(1 / 7)),
                 value_old=jnp.zeros(4), adv=jnp.array([1.0, -1.0, 0.5, -0.5]), target=jnp.zeros(4))
    l, aux = loss(p, batch, 0.2, 0.01, 0.5)
    assert bool(jnp.isfinite(l)) and abs(float(aux["entropy"]) - np.log(7)) < 1e-5


@pytest.mark.slow
def test_make_train_two_updates(conn):
    from flycraftax.brain import BrainParams
    from flycraftax.drive import build_drive
    from flycraftax.env import make_env
    from flycraftax.loop import build_agent
    from flycraftax.ppo import PPOConfig, make_train
    from flycraftax.readout import Readout, readout_groups, std_floor
    from flycraftax.retina import build_retina
    drive = build_drive(conn, build_retina(conn), 100.0, lamina_mv=0.04)
    groups = readout_groups(conn)
    agent = build_agent(conn, drive, Readout(groups, np.zeros(6, np.float32), std_floor(groups)), BrainParams())
    cfg = PPOConfig(n_envs=2, n_steps=4, n_updates=2, n_minibatches=2)
    params, metrics = make_train(agent, make_env(2), cfg)(jax.random.PRNGKey(0))
    assert params["W"].shape == (1314, 7)
    assert metrics["loss"].shape == (2,) and bool(jnp.isfinite(metrics["loss"]).all())
    assert metrics["action_hist"].shape == (2, 7)
