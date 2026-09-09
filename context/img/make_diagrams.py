"""Generate explanatory diagrams for context/bio_background_draft.md.

Run: conda run -n flycraftax python context/img/make_diagrams.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = __file__.rsplit("/", 1)[0]

# ---------------------------------------------------------------- LIF trace

def lif_trace():
    """Two neurons with exact Shiu et al. 2024 parameters.
    Neuron A is a Poisson source at 100 Hz. Neuron B receives A via a 25-synapse
    edge (25 * 0.275 = 6.9 mV per spike), so two closely spaced input spikes
    are needed to cross the 7 mV gap to threshold."""
    rng = np.random.default_rng(3)
    dt, T = 0.1, 300.0                  # ms
    n = int(T / dt)
    v_rest, v_th, t_mbr, tau, t_rfc, t_dly = -52.0, -45.0, 20.0, 5.0, 2.2, 1.8
    w = 25 * 0.275                      # 25 synapses, 6.9 mV per spike
    a, b = np.exp(-dt / tau), np.exp(-dt / t_mbr)
    c = (tau / (t_mbr - tau)) * (b - a)

    t = np.arange(n) * dt
    spikes_a = rng.random(n) < 100 * dt / 1000.0
    v = np.full(n, v_rest); g = np.zeros(n); spikes_b = np.zeros(n, bool)
    refrac = 0.0; delay_steps = int(t_dly / dt)
    for k in range(1, n):
        arriving = spikes_a[k - delay_steps] if k >= delay_steps else False
        if refrac > 0:
            v[k] = v_rest; g[k] = 0.0; refrac -= dt
        else:
            v[k] = v_rest + b * (v[k - 1] - v_rest) + c * g[k - 1]
            g[k] = a * g[k - 1]
        if arriving and refrac <= 0:
            g[k] += w
        if v[k] > v_th:
            spikes_b[k] = True; v[k] = v_rest; g[k] = 0.0; refrac = t_rfc

    fig, ax = plt.subplots(3, 1, figsize=(9, 5.5), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.2, 2]})
    ax[0].vlines(t[spikes_a], 0, 1, color="tab:orange")
    ax[0].set_yticks([]); ax[0].set_ylabel("A spikes\n(input, 100 Hz)")
    ax[1].plot(t, g, color="tab:green")
    ax[1].set_ylabel("B input g\n(mV)")
    ax[2].plot(t, v, color="tab:blue")
    ax[2].axhline(v_th, ls="--", color="grey", lw=1); ax[2].axhline(v_rest, ls=":", color="grey", lw=1)
    ax[2].text(T + 2, v_th, "threshold -45", va="center", fontsize=8)
    ax[2].text(T + 2, v_rest, "rest -52", va="center", fontsize=8)
    for s in t[spikes_b]:
        ax[2].annotate("", (s, -44.5), (s, -42.5), arrowprops=dict(arrowstyle="->", color="red"))
    ax[2].set_ylim(-53, -42); ax[2].set_ylabel("B voltage v\n(mV)"); ax[2].set_xlabel("time (ms)")
    ax[2].set_xlim(0, T + 40)
    fig.suptitle("Leaky integrate-and-fire, Shiu et al. parameters. A drives B through a 25-synapse edge "
                 "(6.9 mV per spike). Red arrows: B spikes.", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/00_lif_trace.png", dpi=150)


if __name__ == "__main__":
    lif_trace(); print("ok")
