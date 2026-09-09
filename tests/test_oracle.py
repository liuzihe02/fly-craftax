import numpy as np

from flycraftax.brain import BrainParams
from flycraftax.oracle import run_brian2


def test_single_kick_fires_next_step():
    p = BrainParams()
    kicks = np.zeros((50, 1), dtype=bool)
    kicks[10, 0] = True
    spikes = run_brian2(1, np.array([], int), np.array([], int), np.array([], float), kicks, 5.0, p)
    assert spikes.shape == (50, 1)
    assert spikes[:, 0].sum() == 1
    assert spikes[11, 0]


def test_strong_edge_propagates_with_delay():
    # neuron 0 kicked once; a big edge onto neuron 1 fires it after the delay.
    # A synapse raises g, not v, and only ~0.49% of a g jump reaches v in one dt
    # (c = 0.00494 of BrainParams.decay), so ~5155 synapses are needed to cross
    # threshold on the first update after arrival. The brief's 30 synapses give a
    # 1.3 mV peak and never fire neuron 1 at all; 6000 is the measured behaviour.
    p = BrainParams()
    kicks = np.zeros((100, 2), dtype=bool)
    kicks[10, 0] = True
    spikes = run_brian2(
        2, np.array([0]), np.array([1]), np.array([6000 * p.w_syn]), kicks, 10.0, p
    )
    t0 = np.flatnonzero(spikes[:, 0])
    t1 = np.flatnonzero(spikes[:, 1])
    assert len(t0) == 1 and len(t1) == 1
    assert t1[0] - t0[0] == p.n_dly + 1  # Brian2 measured: 19 (arrives at t0 + 18, crosses at the next update)


def test_driven_neuron_fires_every_other_step():
    # rfc = 0 for driven neurons, but Brian2 applies resets after synapses, so a
    # kick landing on a spike step is wiped: measured behaviour is alternating spikes.
    p = BrainParams()
    kicks = np.zeros((60, 1), dtype=bool)
    kicks[10:20, 0] = True
    spikes = run_brian2(1, np.array([], int), np.array([], int), np.array([], float), kicks, 6.0, p)
    assert np.flatnonzero(spikes[:, 0]).tolist() == [11, 13, 15, 17, 19]
