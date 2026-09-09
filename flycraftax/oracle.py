"""Brian2 implementation of the Shiu LIF model. Test dependency only."""
import numpy as np

from flycraftax.brain import BrainParams

EQS = """
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau               : volt (unless refractory)
rfc                            : second
"""


def run_brian2(n, pre, post, w_mv, kicks, t_ms, p=None):
    import brian2 as b2

    b2.prefs.codegen.target = "numpy"
    p = p or BrainParams()
    b2.start_scope()
    b2.defaultclock.dt = p.dt_ms * b2.ms
    ns = dict(v_0=p.v_rest * b2.mV, t_mbr=p.t_mbr * b2.ms, tau=p.tau * b2.ms, v_th=p.v_th * b2.mV)
    neu = b2.NeuronGroup(
        n, EQS, method="linear", threshold="v > v_th",
        reset=f"v = {p.v_reset}*mV; g = 0*mV", refractory="rfc", namespace=ns,
    )
    neu.v = p.v_rest * b2.mV
    neu.g = 0 * b2.mV
    neu.rfc = p.t_rfc * b2.ms
    driven = np.flatnonzero(kicks.any(axis=0))
    neu.rfc[driven] = 0 * b2.ms

    objs = [neu]
    if len(pre):
        syn = b2.Synapses(neu, neu, "w : volt", on_pre="g += w", delay=p.t_dly * b2.ms)
        syn.connect(i=np.asarray(pre), j=np.asarray(post))
        syn.w = np.asarray(w_mv) * b2.mV
        objs.append(syn)

    t_idx, n_idx = np.nonzero(kicks)
    if len(t_idx):
        gen = b2.SpikeGeneratorGroup(n, n_idx, t_idx * p.dt_ms * b2.ms)
        kick = b2.Synapses(gen, neu, on_pre=f"v += {p.kick}*mV")
        kick.connect(j="i")
        objs += [gen, kick]

    mon = b2.SpikeMonitor(neu)
    net = b2.Network(*objs, mon)
    net.run(t_ms * b2.ms)

    T = round(t_ms / p.dt_ms)
    out = np.zeros((T, n), dtype=bool)
    steps = np.round(mon.t / b2.defaultclock.dt).astype(int)
    out[steps[steps < T], mon.i[steps < T]] = True
    return out
