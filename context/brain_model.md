# Brain dynamics model (Shiu et al. 2024 LIF)

Reference notes for porting the Shiu et al. whole-brain leaky integrate-and-fire (LIF) model to JAX.

Sources used:
- Nature paper: Shiu, P.K. et al. "A Drosophila computational brain model reveals sensorimotor processing", *Nature* 634, 210-219 (2024). https://www.nature.com/articles/s41586-024-07763-9
  - Open-access full text (used here): https://europepmc.org/articles/PMC11446845 , XML via `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11446845/fullTextXML`
- bioRxiv preprint (earlier, same model): https://doi.org/10.1101/2023.05.02.539144 , PDF pages 21-25 are the Methods
- Author code: https://github.com/philshiu/Drosophila_brain_model (MIT)
- Local vendored copy + multi-backend harness: `/home/flowingpurplecrane/personal/fly-craftax/external/fly-brain/`
- Modelling results archive: https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.CZODIW

---

## Core LIF equations

### State variables

- Two state variables per neuron, both in volts.
  - `v` - membrane potential.
  - `g` - synaptic "conductance" term; despite the name it has units of **volts**, not siemens.
- Plus a per-neuron refractory-period value `rfc` (seconds), which the code mutates per neuron (set to 0 for Poisson-driven neurons).

### Membrane and synapse ODEs

Nature Methods, "Computational model" (three equations, one of them the jump rule):

$$
\begin{aligned}
\frac{dv_i}{dt} &= \frac{g_i - (v_i - V_{\text{resting}})}{T_{\text{mbr}}} \\[4pt]
\frac{dg_i}{dt} &= -\frac{g_i}{\tau} \\[4pt]
g_i &\leftarrow g_i + w_{j,i} \quad \text{upon spike from neuron } j
\end{aligned}
$$

- Both ODEs are frozen during the refractory period (Brian2 `(unless refractory)` flag on both lines).
- This is a **current-based** synapse in disguise: `g` is added directly into the numerator of the `dv/dt` equation in volts, so there is no reversal potential and no voltage dependence. Excitation and inhibition have identical driving force.
- The paper calls it an "alpha synapse" but the implementation is a **single exponential** on `g`, not a true alpha (difference-of-exponentials) function. The `w` jump is instantaneous, then decays with $\tau$. Treat the paper's "$\alpha$-synapse" wording as loose.

### Threshold, reset, refractory

- Threshold condition: `v > v_th` (strict inequality).
- On spike, the reset applies **three** things (`model.py:55`):
  - `v <- v_rst`
  - `g <- 0 mV` (the conductance is wiped, not just the voltage)
  - `w = 0` - a no-op. `w` is a `Synapses` variable, not a `NeuronGroup` variable; inside Brian2 reset abstract code this just creates a discarded temporary. Do not port it.
- Refractory: after a spike, both `v` and `g` are held frozen for `t_rfc`. This caps the firing rate at $1/2.2\ \text{ms} \approx 454$ Hz.
- Poisson-driven neurons have their refractory period set to **0** (`model.py:95, 106`), so their rate is unclamped and tracks the drive rate.

### Delay

- One uniform axonal delay `t_dly = 1.8 ms` on every synapse (`model.py:177`, `delay=params['t_dly']`).
- Because it is uniform and the weights are static, delaying the **spike vector** is exactly equivalent to delaying the synaptic current. Important for the port (see JAX section).

### Noise

- No intrinsic noise, no background current, no membrane noise.
- The **only** stochasticity is the Poisson external drive. Baseline firing of every unstimulated neuron is exactly 0 Hz.
- Consequence stated explicitly in the paper: inhibitory input onto an inactive neuron has zero effect. Inhibition is only visible where there is already activity.

### Synapse count to weight

- Linear, no saturation, no normalization by in-degree:

$$
w_{j,i} = s_j \cdot n_{j,i} \cdot W_{\text{syn}}
$$

  - $n_{j,i}$ = raw synapse count from neuron $j$ onto neuron $i$ (integer, from the connectome).
  - $s_j \in \{+1, -1\}$ = sign of the **presynaptic** neuron (excitatory / inhibitory). Sign is a property of the neuron, not the edge.
  - $W_{\text{syn}} = 0.275$ mV per synapse.
- In the code the product $s_j \cdot n_{j,i}$ is precomputed in the data file as the column `Excitatory x Connectivity`, and multiplied by `w_syn` at build time (`model.py:183`).
- Excitatory and inhibitory synapses have the **same magnitude** per synapse. The paper tested +/-50% on the I:E ratio and found predictions robust (95-96% consistent with default).

### Numerical parameters

| Symbol (paper) | Code key | Value | Units | Source cited by paper |
|---|---|---|---|---|
| $V_{\text{resting}}$ | `v_0` | -52 | mV | Kakaria & de Bivort 2017 |
| $V_{\text{reset}}$ | `v_rst` | -52 | mV | Kakaria & de Bivort 2017 |
| $V_{\text{threshold}}$ | `v_th` | -45 | mV | Kakaria & de Bivort 2017 |
| $R_{\text{mbr}}$ | (implicit) | 10 kOhm cm^2 (preprint says 10 MOhm) | - | Kakaria & de Bivort 2017 |
| $C_{\text{mbr}}$ | (implicit) | 2 uF cm^-2 (preprint says 0.002 uF) | - | Kakaria & de Bivort 2017 |
| $T_{\text{mbr}}$ | `t_mbr` | **20** | ms | $= C_{\text{mbr}} \times R_{\text{mbr}}$ |
| $\tau$ | `tau` | 5 | ms | Juergensen et al. 2021 |
| $T_{\text{refractory}}$ | `t_rfc` | 2.2 | ms | Kakaria & de Bivort 2017; Lazar et al. 2021 |
| $T_{\text{dly}}$ | `t_dly` | 1.8 | ms | Paul et al. 2015 |
| $W_{\text{syn}}$ | `w_syn` | 0.275 | mV per synapse | **free parameter**, fit |
| - | `f_poi` | 250 | dimensionless | Poisson weight scaling |
| - | `r_poi` | 150 (default) | Hz | default Poisson drive rate |
| - | `r_poi2` | 0 (default) | Hz | second Poisson class |
| - | `t_run` | 1000 | ms | trial duration |
| - | `n_run` | 30 | - | trials per experiment |
| - | `dt` | 0.1 | ms | Brian2 `defaultclock.dt` (default, never overridden) |

- Note the unit mismatch on $R_{\text{mbr}}$ / $C_{\text{mbr}}$ between preprint and Nature version. It does not matter: only their product $T_{\text{mbr}} = 20$ ms enters the equations, and both versions agree on 20 ms.
- Driving voltage range is small: $V_{th} - V_{rest} = 7$ mV. A single synapse contributes 0.275 mV, so roughly 25 coincident synapses (from a single 25-synapse edge, or many small edges) are needed to fire a resting neuron.
- $W_{\text{syn}}$ was fit so that "activation of sugar GRNs at 100 Hz resulted in roughly 80% of maximal MN9 firing". Robustness: +/-30% on $W_{\text{syn}}$ leaves 90-95% of the 164 testable predictions unchanged.

### Simulator and integration

- **Brian2** (Stimberg, Brette & Goodman 2019), `method='linear'` (`model.py:166`).
  - `method='linear'` means Brian2 solves the linear system **exactly** over each `dt` via matrix exponential. It is NOT forward Euler.
- `dt = 0.1 ms`, never set explicitly - it is Brian2's default clock.
- 30 trials of 1000 ms per experiment, run in parallel across CPU cores with joblib/loky (`model.py:355-361`).
- Paper reports ~5 min per 1000 ms trial per CPU thread for the sugar experiment.

### Exact update (what `method='linear'` computes)

Worth porting directly, because it is stable at any `dt` and matches the reference exactly. With $u = v - V_{\text{rest}}$:

$$
\begin{aligned}
g(t+\Delta t) &= g(t)\, e^{-\Delta t/\tau} \\[4pt]
u(t+\Delta t) &= u(t)\, e^{-\Delta t/T_{\text{mbr}}} + g(t)\,\frac{\tau}{T_{\text{mbr}} - \tau}\left(e^{-\Delta t/T_{\text{mbr}}} - e^{-\Delta t/\tau}\right)
\end{aligned}
$$

- With $\tau = 5$, $T_{\text{mbr}} = 20$: the coupling coefficient $\tau/(T_{\text{mbr}} - \tau) = 1/3$, and $e^{-\Delta t/20} > e^{-\Delta t/5}$, so positive `g` depolarizes as expected.
- Precompute three scalars once: $a = e^{-\Delta t/\tau}$, $b = e^{-\Delta t/T_{\text{mbr}}}$, $c = \tfrac{1}{3}(b - a)$. Then the whole step is `g_new = a*g`, `v_new = v_rest + b*(v - v_rest) + c*g`.
- Order matters: use the **old** `g` in the `v` update, then decay `g`, then add the new synaptic jumps.

---

## Neurotransmitter signing

Nature Methods, "Neurotransmitter predictions".

- Predictions come from Eckstein et al. 2024 (`Cell`), which predicts a neurotransmitter **per synapse**, not per neuron.
- Rules, verbatim in effect:
  - **Cleft score cutoff of 50** on each presynaptic site (as in Baker et al. 2022). Sites below this are discarded.
  - For each surviving presynaptic site, take the **argmax** neurotransmitter prediction.
  - If **more than half** of all presynaptic sites across the whole neuron are predicted inhibitory (GABA or glutamate), the neuron is assigned **inhibitory**; otherwise **excitatory**.
  - So it is a per-neuron majority vote over sites, not a per-site confidence threshold, and not a per-edge decision.
- Sign assignment per neurotransmitter:

| NT | Sign in model |
|---|---|
| ACh (acetylcholine) | excitatory (+1) |
| GABA | inhibitory (-1) |
| Glutamate | **inhibitory** (-1) |
| Dopamine | excitatory (+1) |
| Octopamine | excitatory (+1) |
| Serotonin (5-HT) | excitatory (+1) |
| unknown / no prediction | falls out of the majority vote; effectively excitatory unless the inhibitory sites win |

- Neuromodulators (DA, OA, 5-HT) get **no special treatment at all**. They are lumped into the excitatory bucket and fire as fast excitatory synapses. The paper flags this as a known weakness ("other neurons (for example, dopaminergic or serotonergic) will be modelled less well").
- Neuropeptides and volume transmission are not modelled at all.
- Brain-wide NT composition quoted by the paper (from Eckstein et al.): **~55% cholinergic, 24% glutamatergic, 14% GABAergic, 7% DA/OA/5-HT.**
  - So roughly 38% of neurons end up inhibitory brain-wide. Measured in the shipped v783 data: **30.0% of presynaptic neurons are -1** (see below).
- Glutamate-is-inhibitory is the single most consequential assumption. The paper ran the counterfactual:
  - Assuming glutamate excitatory **eliminates** the finding that bitter and Ir94e GRNs are inhibitory.
  - It raises the false-positive rate of the optogenetic activation experiment from **1% to 16%**.
  - Keep glutamate inhibitory unless you deliberately want to test this.

---

## Edge thresholding

**The paper applies no synapse-count threshold.** This is a surprise given the task framing, and worth stating plainly.

- The Nature text never mentions a minimum synapse count. Searched the full text for "threshold" - all seven hits are the firing threshold, the skeletonization threshold, or the 80%-of-control silencing criterion.
- Verified empirically against the shipped connectivity files in `external/fly-brain/data/`:

| Dataset | Neurons | Edges | Total synapses | min `Connectivity` |
|---|---|---|---|---|
| FlyWire v630 (paper) | 127,400 modelled (127,015 with outputs) | 14,687,178 | 52,793,639 | **1** |
| FlyWire v783 (repo default) | 138,639 | 15,091,983 | 54,492,922 | **1** |

- So every single-synapse edge is retained. A 1-synapse edge contributes 0.275 mV, ~4% of the 7 mV to threshold - individually negligible but there are ~7.5M of them.
- What a `>= 5` threshold would do, measured on the same data (both versions give near-identical ratios):

| Threshold | Edges kept (v783) | % of edges | % of total synaptic mass kept |
|---|---|---|---|
| `>= 1` | 15,091,983 | 100.0% | 100.0% |
| `>= 2` | 7,595,967 | 50.3% | 86.2% |
| `>= 3` | 4,916,231 | 32.6% | 76.4% |
| `>= 5` | **2,700,513** | **17.9%** | **62.7%** |
| `>= 10` | 1,066,822 | 7.1% | 43.4% |

- Reading: a `>= 5` cut discards 82% of edges but only 37% of synaptic mass. That is a large but not absurd perturbation. Expect systematically lower downstream firing rates than the reference.
- The paper gives **no** data on how thresholding changes results - they never tried it. The closest robustness experiment is the weight shuffle: with correct connectivity, 100 Hz sugar activation drives MN9 in 100% of simulations; with shuffled weights (distribution preserved), 1 of 100.
- Practical implication for a `>= 5` port: you will likely need to raise $W_{\text{syn}}$ (or the input drive) to recover comparable MN9-equivalent firing. The paper's own robustness test shows +30% on $W_{\text{syn}}$ keeps 95% of predictions, so there is headroom to compensate. Recalibrate against a known input-output pair rather than keeping 0.275 mV blindly.

---

## Stimulation and readout protocol

### Activation

- Implemented as Brian2 `PoissonInput` writing **directly into `v`**, not into `g` (`model.py:88-96`):

```python
p = PoissonInput(
    target=neu[i],
    target_var='v',
    N=1,
    rate=params['r_poi'],
    weight=params['w_syn']*params['f_poi']
    )
neu[i].rfc = 0 * ms
```

- Per driven neuron, per timestep: draw `n ~ Binomial(N=1, p=rate*dt)` (i.e. Bernoulli), then `v += n * w_syn * f_poi`.
- Effective kick: `0.275 mV * 250 = 68.75 mV`. The gap to threshold is only 7 mV, so **every Poisson event fires the neuron immediately**, in the same timestep.
- Combined with `rfc = 0` on those neurons, a driven neuron's output rate equals the Poisson rate almost exactly. Confirmed against the shipped reference outputs: 200 Hz drive gives GRN rates of 197-202 Hz, 100 Hz drive gives 99-102 Hz.
- So conceptually this is **not** stochastic current injection - it is "make this neuron a Poisson spike source at rate r". Port it that way: it is cheaper and equivalent.
- Two independent drive classes are supported (`r_poi`, `r_poi2`) for co-activation experiments (e.g. sugar + bitter).
- Note `neu_exc` is a flat list of neuron IDs, not cell types. Bilateral / multi-neuron cell types are activated by listing all their IDs.

### Silencing

- Zero the **outgoing** weights of the silenced neuron (`model.py:127-128`):

```python
for i in slnc:
    syn.w[' {} == i'.format(i)] = 0*mV
```

- Here `i` is Brian2's presynaptic index variable, so this zeroes row `i` of the outgoing weight matrix only.
- Matches the paper ("silenced by eliminating all output of that neuron"). The `fly-brain` README claim that it sets connections "to and from those neurons to zero" is **wrong** - inputs are untouched.
- The silenced neuron still spikes; it just has no downstream effect.

### Simulation length and readout

- 30 trials x 1000 ms per condition, fresh network per trial (`run_trial` rebuilds the model each time - `model.py:278`).
- Readout is dead simple: **total spike count over the whole 1000 ms trial divided by 1 s, then averaged across the 30 trials** (`utils.py:262-267`). No windowing, no binning, no transient discard, no steady-state detection.
- "Activated neuron" is defined as **firing rate > 0 Hz** (any spike in any trial).
- Silencing phenotype criterion: MN9 firing drops to **<= 80% of unsilenced control** at any tested drive frequency.
- Frequency sweeps used:
  - Sugar GRNs: 10-200 Hz in 10 Hz steps.
  - Water GRNs: 20-260 Hz.
  - JONs (grooming): 20-220 Hz.
  - Silencing screens: 50-120 Hz in 10 Hz steps (8 frequencies).
  - Secondary "which neuron activates MN9" screens: top 200 responders driven at 25, 50, 75, 100, 125, 150, 175, 200 Hz.
- Laterality caveat: FAFB/FlyWire was found to be left-right inverted. The Nature version says "unilateral **left** hemisphere activation for all simulations"; the bioRxiv version says right. Same neurons, corrected labelling. Use the flywire IDs, not the side names.

---

## Validated results usable as sanity checks

### Concrete reference firing rates (computed from shipped outputs)

Computed directly from `external/../shiu_repo/results/example/*.parquet` (author-generated, FlyWire v630, 30 trials x 1 s). These are the strongest unit-test targets available because they are exact model outputs, not experimental claims.

- MN9 flywire ID (v630): **`720575940660219265`** (from `figures.ipynb`, Figure 1E cell).
- Sugar GRN set: the 21 IDs listed in `example.ipynb` (also duplicated in `external/fly-brain/code/benchmark.py:61-83`, with one ID swapped - see Deviations).

| Experiment | Drive | MN9 mean rate | Active neurons | Total spikes (30 trials) |
|---|---|---|---|---|
| `sugarR` | sugar GRNs @ 200 Hz | **93.27 Hz** | 448 | 511,566 |
| `sugarR_100Hz` | sugar GRNs @ 100 Hz | **67.03 Hz** | 404 | 289,073 |
| `sugarR` + silence `720575940617937543` | 100 Hz | 63.27 Hz | 414 | 277,853 |
| `sugarR` + silence `720575940621754367` | 100 Hz | 66.90 Hz | 402 | 287,004 |
| `sugarR` + silence `720575940622695448` | 100 Hz | 71.17 Hz | 413 | 300,963 |

- Driven GRNs themselves come back at 197-202 Hz (200 Hz drive) and 99-102 Hz (100 Hz drive) - a direct check that your Poisson injection is calibrated.
- MN9 at 93.27 Hz for 200 Hz drive vs 67.03 Hz for 100 Hz drive is consistent with the "$W_{syn}$ chosen so 100 Hz sugar gives ~80% of maximal MN9" fit.
- Note the silencing deltas are small (63-71 Hz vs 67 Hz control) - these three are not strong phenotypes; use them as regression checks, not as behavioural claims.

### Paper-level counts (good integration tests)

- **Sugar GRN activation drives MN9** (Fig. 1c). Also drives MN6, MN8, MN11. MN9 and MN11 confirmed sugar-responsive in vivo.
- **Contralateral bias**: unilateral sugar GRN activation drives the *contralateral* MN9 more than the ipsilateral, from either hemisphere (Fig. 1c, Extended Data Fig. 1d). A cheap and discriminating structural test.
- **Response set size scales with drive** (Supplementary Table 1): of 127,400 neurons, **45** respond at 10 Hz sugar drive, **455** at 200 Hz.
- **Sugar/water overlap** (Fig. 3f): at the drive level where each pathway produces 40 Hz MN9 firing, sugar activates **377** neurons, water activates **391**, with **250** shared.
- **Aversive segregation**: at minimal drive that reduces 40 Hz MN9 to 1 Hz, only **2** neurons shared between sugar and bitter, **30** between sugar and Ir94e.
- **Bitter and Ir94e inhibit MN9**. Strong bitter eliminates MN9 firing at strong sugar; strong Ir94e does not fully eliminate it. Both confirmed behaviourally.
- **Silencing screen** (Fig. 1h): 47 neurons sugar-responsive and sufficient for MN9 activation; **14** of those also required (>20% MN9 drop when silenced). For water: **39** sufficient, of which **30** also sugar-activated, and **9** both necessary and sufficient.
- **Optogenetic screen** (Fig. 2): of 106 SEZ split-GAL4 cell types driven at 50 Hz, **11** predicted to activate MN9, **10 of 11** confirmed. Of the 95 predicted negative, only **4** false negatives. >90% accuracy.
- **Overall**: 164 testable predictions, **91%** consistent with experiment; **84%** excluding the optogenetic screen.
- **Grooming circuit** (Fig. 5): JON activation at 20-220 Hz. Only **4** neurons besides aDN1 itself can elicit aDN1 activity (aBN1, aDN2, and two others below 2 Hz). Only **3** besides aDN1 reduce aDN1 activity >20% at 140 Hz JON drive (aBN1, a BN2-class descending neuron, aDN2).
- **Known false negatives** to expect: Usnea and Phantom are predicted inhibitory and therefore fail to activate MN9 in the model, but do drive proboscis extension optogenetically. This is the zero-baseline-rate limitation biting.

### DNp09, MDN, DNa02, dFB, NPF - not in this paper

Important correction to the task framing:

- **None of DNp09, MDN, DNa02, dFB, or NPF appear anywhere in the Shiu et al. Nature paper.** Confirmed by full-text search (0 hits each).
- The paper explicitly avoids descending-neuron circuits: "other sensorimotor circuits ... require descending neurons, which are incomplete in the Flywire volume."
- The only descending neurons validated are the **grooming** aDN1 / aDN2 / aBN1 / aBN2 set.
- Those DN names come from **desktop-fly** (https://github.com/DenisSergeevitch/desktop-fly), a hobby project that also uses FlyWire + MaleCNS with a LIF model. Its behavioural mapping is:
  - DNp01 (Giant Fiber) - escape takeoff
  - DNp09 - forward walking (rate through a modelled state threshold)
  - MDN - backward walking
  - DNa01 + DNa02 - steering
  - DNg11 - grooming
  - LC4 / LPLC2 - looming detectors feeding the escape pathway
- That mapping is a **modelling choice, not a validated model output**. It rests on the optogenetics literature for those DNs, not on any LIF simulation result. Do not use it as a sanity-check target for a port; use it as a design template for action decoding.
- desktop-fly's own scale, for calibration: 668-neuron FlyWire circuit with 18,968 connection rows, plus a 1,045-neuron MaleCNS locomotor circuit with 17,224 connections (708,689 synaptic contacts), run at **1 kHz** (`dt = 1 ms`).

---

## The `fly-brain` repo

Path: `/home/flowingpurplecrane/personal/fly-craftax/external/fly-brain/`. Origin `https://github.com/eonsystemspbc/fly-brain.git`, single commit `a3db62f`.

### What it actually is

- Primarily a **multi-backend benchmark harness**, not a new model. It re-implements the identical Shiu model across six simulators and measures wall-clock time and cross-backend spike agreement.
- The scientific model itself is the vendored, near-verbatim copy of the author code.

### Layout

| Path | Role |
|---|---|
| `code/paper-phil-drosophila/model.py` | Vendored Shiu Brian2 model - the canonical reference |
| `code/paper-phil-drosophila/utils.py` | `load_exps`, `get_rate` (rate = spikes / t_run, averaged over trials) |
| `code/benchmark.py` | Config, paths, `EXPERIMENTS` dict, logging, dispatcher |
| `code/run_pytorch.py` | PyTorch/CUDA re-implementation - **the closest analogue to a JAX port** |
| `code/run_brian2_cuda.py` | Brian2 C++ standalone and Brian2CUDA |
| `code/run_genn.py`, `run_nestgpu.py`, `run_brian2_genn.py` | GeNN, NEST GPU, Brian2GeNN |
| `code/compare_ground_truth.py` | Jaccard on active-neuron sets, Pearson on per-neuron rates, spike-count ratio |
| `data/2025_Completeness_783.csv` | Neuron list, 138,639 rows, one column `Completed` |
| `data/2025_Connectivity_783.parquet` | 15,091,983 edges |
| `data/archive/2023_*_630.*` | v630 - the version the paper used |

### Connectome data

- Default is **FlyWire v783**, not the paper's v630. Both are shipped. **Neither is MaleCNS.** The repo has no MaleCNS support at all; your MaleCNS v1.0 ingestion is entirely new work.
- `Completeness` CSV: index = flywire ID, one boolean column. The **row order defines the neuron index** - `flyid2i = {j: i for i, j in enumerate(df_comp.index)}`. Fragile: any reordering silently invalidates cached weight matrices.
- `Connectivity` parquet columns:
  - `Presynaptic_ID`, `Postsynaptic_ID` (int64 flywire IDs)
  - `Presynaptic_Index`, `Postsynaptic_Index` (int64 row indices into the completeness CSV)
  - `Connectivity` (int, raw synapse count, min 1, max 2405)
  - `Excitatory` (int, +1 or -1 only - no zeros, no unknowns)
  - `Excitatory x Connectivity` (int, the signed product, range -2405 to +1897) - this is the only column the model reads
- The signing is **pre-baked into the data file**. The NT prediction pipeline is not in this repo; you cannot re-derive or re-threshold the signs from what is shipped.

### Sparse weight storage (PyTorch backend)

`run_pytorch.py:280-322`. Builds a COO tensor then converts to CSR, pickling both to `data/weight_coo.pkl` (~288 MB) and `weight_csr.pkl` (~289 MB).

```python
idx = [
    data_conn['Postsynaptic_Index'].to_list(),
    data_conn['Presynaptic_Index'].to_list(),
]
val = data_conn['Excitatory x Connectivity'].to_list()
weight_coo = torch.sparse_coo_tensor(
    idx, val, (num_neurons, num_neurons)
).to(torch.float32)
```

- Note the index order: **(post, pre)**, so `W[i, j]` is the weight from `j` onto `i`. The forward pass then does `spikes @ W.T`.
- Values are the raw signed synapse **counts**; `w_scale = 0.275` is applied later at the matmul site, not baked into the matrix. Convenient - you can rescale $W_{\text{syn}}$ without rebuilding the sparse structure.

### Core step function (PyTorch backend)

The clearest statement of one timestep, `run_pytorch.py:247-268`:

```python
def forward(self, rates, conductance, delay_buffer, spikes, v, refrac, generator=None):
    poisson_spikes = self.poisson(
        rates,
        generator=generator
    )

    voltage_stim = self.scale * poisson_spikes

    weighted_spikes = torch.matmul(
        spikes,
        self.weights.transpose(0, 1)
    )

    recurrent_input = self.scale * weighted_spikes

    conductance, delay_buffer, spikes, v, refrac = self.neurons(
        recurrent_input,
        voltage_stim,
        conductance,
        delay_buffer,
        spikes,
        v,
        refrac,
    )
    return conductance, delay_buffer, spikes, v, refrac
```

Poisson generation, `run_pytorch.py:63-64`:

```python
def forward(self, rates, generator=None):
    return torch.bernoulli(rates * self.prob_scale, generator=generator) * self.scale
```

where `prob_scale = dt/1000` and `scale = 250`.

Membrane update, `run_pytorch.py:118-129` - note this is **forward Euler**, not Brian2's exact linear solve:

```python
v = v + voltage_stim
v = v + self.time_factor * (conductance - (v - self.v_rest))

spike = self.spike_gradient(v - self.v_threshold)

reset = ((v - self.v_reset) * spike).detach()
v = v - reset
```

Synapse update with the delay ring buffer, `run_pytorch.py:86-91`:

```python
conductance_new = (
    conductance * (1 - self.time_factor) + delay_buffer[:, 0, :] * refrac
)
delay_buffer = torch.roll(delay_buffer, shifts=-1, dims=1)
delay_buffer[:, -1, :] = input_
```

- Delay buffer shape is `(batch, steps_delay + 1, N)` with `steps_delay = int(1.8 / 0.1) = 18`, i.e. 19 slots. It buffers the **weighted current**, which is 32x more memory than buffering the spike bits would be.

### Activate / silence API

- Two levels.
  - Paper code: `run_exp(exp_name, neu_exc, path_res, path_comp, path_con, params=..., neu_slnc=[], neu_exc2=[], n_proc=-1)`. Takes flywire IDs, handles ID-to-index mapping internally, runs `n_run` trials in parallel, writes a parquet of `(t, trial, flywire_id, exp_name)`.
  - Benchmark harness: a static `EXPERIMENTS` dict (`benchmark.py:57-101`) with `neu_exc`, `neu_exc2`, `neu_slnc`, `stim_rate`. Two experiments only: `sugar` (21 GRNs @ 200 Hz) and `p9` (2 P9 neurons @ 100 Hz - `720575940627652358` left, `720575940635872101` right).
- **The benchmark backends do not implement silencing at all.** `neu_slnc` is present in the config dict but always empty and never consumed by `run_pytorch.py`. Only the vendored Brian2 `model.py` can silence.
- No streaming / stepping API anywhere. Every backend runs a fixed-duration trial to completion and dumps spikes. There is no notion of injecting time-varying input mid-run. For RL you need a step-wise driver that none of these provide.

### Performance notes (their numbers, RTX 4070, 8 GB class)

From `data/benchmark-results.csv`, sugar experiment, means across 5 rounds:

| Backend | `t_run=1s`, `n_run=1` sim time | realtime ratio | `n_run=32` sim time | realtime ratio |
|---|---|---|---|---|
| GeNN (CUDA) | 0.56 s | 1.82x | 24.9 s | 1.49x |
| NEST GPU | 1.09 s | 0.93x | 40.4 s | 0.86x |
| Brian2GeNN | 1.81 s | 0.57x | 60.1 s | 0.55x |
| Brian2 (CPU) | 2.89 s | 0.35x | 72.4 s | 0.45x |
| PyTorch (CUDA) | 9.38 s | 0.11x | 184.1 s | 0.17x |
| Brian2CUDA | 12.15 s | 0.08x | 396.9 s | 0.08x |

- Read this as a warning. The dense-ish PyTorch approach is **~0.94 ms per 0.1 ms timestep** for 138k neurons and 15M edges. Only GeNN beats realtime, and only barely.
- Consistency across backends: ~14,000 spikes and ~320-340 active neurons per 1 s trial regardless of backend. The comparison script's pass criteria are Pearson > 0.99 with Jaccard > 0.90 (excellent) or > 0.95 / > 0.80 (good).
- All six backends use identical parameter values (verified: `run_genn.py:35-47`, `run_nestgpu.py:44-52`, `run_brian2_cuda.py:40-53` all carry -52/-45/20/5/2.2/1.8/0.275/250 and `DT = 0.1`).

### License

- Repo overall: **GPL-2.0-or-later**. This is viral - if you vendor or derive from `code/`, your project inherits it.
- `code/paper-phil-drosophila/` retains its upstream **MIT** license (Copyright 2023 Philip Shiu and Nico Spiller). Safe to derive from.
- Practical advice: **port from the MIT `model.py` and from the paper**, and treat `run_pytorch.py` as read-only inspiration you do not copy line-for-line, unless you are happy to be GPL.

### Deviations from the paper

- Default connectome is **v783 (138,639 neurons)**, not the paper's v630 (127,400). Results will not match published numbers exactly.
- The PyTorch/GeNN/NEST backends use **forward Euler**, the paper uses Brian2 `method='linear'` (exact). The reported Pearson > 0.99 agreement suggests this is tolerable at `dt = 0.1 ms`, but it is a real difference and will diverge at coarser `dt`.
- Refractory handling differs. Brian2 freezes both `v` and `g` during refractoriness while still accumulating synaptic input into `g`. The PyTorch port never freezes `v` at all, and instead **drops** incoming synaptic input during refractoriness (`delay_buffer[:, 0, :] * refrac`). Different mechanism, similar aggregate effect.
- The `sugar` GRN list in `benchmark.py` differs from `example.ipynb` by one ID (`720575940620900446` in the notebook vs `720575940621754367` in the harness). Minor, but it means benchmark outputs are not bit-comparable to the shipped example outputs.
- `force_overwrite` default flipped from `False` (upstream) to `True` in the vendored `model.py:297`.
- No silencing in any GPU backend, as noted.

---

## Porting notes for JAX

### Representation

- **Dense is out.** 120,000^2 x 4 bytes = 57.6 GB. Not on 8 GB, not on anything.
- **BCOO is the right default.** For E edges with int32 indices and float32 data: 12 bytes/edge.
  - `>= 5` synapses on a FlyWire-scale graph: ~2.7M edges = **32 MB**. Negligible.
  - No threshold: ~15M edges = **180 MB**. Also fine.
  - MaleCNS v1.0 at ~120k neurons and `>= 5` should land in the same 2-5M edge range; budget under 100 MB either way.
- `jax.experimental.sparse.BCOO` with `@jax.jit` gives a fused sparse-dense matmul. Sort indices once (`bcoo_sort_indices`) and set `unique_indices=True` so XLA can pick the fast path.
- `segment_sum` is the alternative: store `(pre_idx, post_idx, w)` flat and do `jax.ops.segment_sum(w * spikes[pre_idx], post_idx, num_segments=N)`. Equivalent cost, more control, and it makes gradient/masking tricks easier. Pick one and benchmark; do not build both.
- Batch over environments as the **dense** dimension: `BCOO (N,N) @ dense (N, B)`. One sparse structure shared across all envs.

### Memory budget at N = 120,000, E = 3M, B envs

| Item | Per env | B = 64 |
|---|---|---|
| `v`, `g` (f32) | 0.96 MB | 61 MB |
| refractory counter (i32) | 0.48 MB | 31 MB |
| spike vector (f32) | 0.48 MB | 31 MB |
| delay buffer as **spikes**, 19 slots (bool) | 2.3 MB | 146 MB |
| delay buffer as **currents**, 19 slots (f32) | 9.1 MB | 583 MB |
| sparse weights (shared) | - | 36 MB |
| readout counters | 0.48 MB | 31 MB |

- **Buffer the spike vector, not the synaptic current.** The delay is uniform 1.8 ms, so `matvec(delay(s)) == delay(matvec(s))`. Buffering spikes is 4-32x cheaper and you do one matvec per step either way.
- Everything fits in well under 1 GB at B = 64 on an 8 GB card. Memory is not your constraint. **Throughput is.**

### Timestep and steps per action

- `dt = 0.1 ms` reproduces the reference exactly. `t_dly / dt = 18` and `t_rfc / dt = 22`, both clean integers.
- `dt = 0.2 ms` also divides cleanly (9 delay steps, 11 refractory steps) and halves the step count.
- `dt = 0.5 ms` does **not** divide 1.8 ms. Avoid unless you round the delay.
- `dt = 1.0 ms` is what desktop-fly uses. The exact-exponential update is unconditionally stable there, but delay collapses to 2 steps and refractory to 2, and spike timing is heavily quantized.
- Recommendation: **implement the exact exponential update** (three precomputed scalars, above) so `dt` is a free knob. Validate at `dt = 0.1 ms` against the reference numbers in the table above, then coarsen to 0.2 or 0.5 ms and measure the rate error before committing.
- Steps per Craftax action: the paper's 1000 ms readout window is far too long. A 20-50 ms window is the practical range - long enough for the 1.8 ms delay to propagate several hops and for rates to be estimable, short enough to be affordable. At `dt = 0.2 ms`, 50 ms = **250 steps per action**.
- Rate estimation noise at short windows is real: a neuron firing at 40 Hz emits 2 spikes in 50 ms. Either use longer windows, average over the env batch, or read out a **population** rather than a single neuron.

### Injecting drive

Reproduce `PoissonInput` exactly:

- Precompute a static `drive_rate` array of shape `(B, N)`, zero except at the driven indices.
- Per step: `stim = jax.random.bernoulli(key, drive_rate * dt / 1000.0)`, then `v = v + 68.75 * stim` (`w_syn * f_poi`).
- Set `refrac_steps = 0` for driven neurons so they are never clamped.
- Because the 68.75 mV kick always exceeds the 7 mV gap to threshold, the cheaper and equivalent implementation is: **force `spike = 1` for driven neurons wherever the Bernoulli fires**, and skip the voltage arithmetic entirely.
- Keep the drive indices **static** (fixed neuron set, varying rates) so the whole step stays jittable with no recompilation. Vary `drive_rate` values per env, never the index set.

### Silencing

- Do **not** mutate the sparse weights. Multiply the spike vector by a static-shaped `silence_mask` of shape `(B, N)` before the matvec: `masked_spikes = spikes * silence_mask`.
- Mathematically identical to zeroing all outgoing weights of the masked neurons, costs one elementwise multiply, and keeps the sparse structure and the jit cache intact.

### Readout

- Never materialize a full raster. `120,000 x 250 steps x 64 envs` booleans is 1.9 GB per action.
- Accumulate inside the `lax.scan` carry: `counts = counts + spikes`, shape `(B, N)` float32 = 31 MB. Gather the readout indices at the end.
- Or, if the readout set is small and fixed, accumulate only `counts_ro = counts_ro + spikes[:, readout_idx]` - shape `(B, K)`, trivial.
- Convert to rate: `rate_hz = counts / (n_steps * dt_ms / 1000.0)`.
- The whole per-action rollout is one `jax.lax.scan` over steps with carry `(v, g, refrac_counter, spike_buffer, counts, key)`.

---

## Gotchas and known limitations

Stated by the paper (Methods, "Computational modelling limitations"), plus observations from the code.

- **Baseline firing is exactly 0 Hz.** The single most distorting assumption. Inhibitory input onto a silent neuron does nothing, so purely inhibitory circuits are invisible until something else drives them. This is why Usnea and Phantom fail as predictions. For an RL agent this means the brain is completely inert until you drive it - there is no ongoing activity, no spontaneous behaviour, no state that persists between actions unless you carry `v` and `g` across action boundaries.
- **Absolute firing rates are not meaningful.** The authors say so explicitly: interpret differences between conditions across a range of drive rates, not single numbers. Any reward or action decoding built on absolute rate thresholds will be fragile.
- **No neuromodulation.** DA, OA and 5-HT are modelled as fast excitatory synapses. No slow modulation of gain, threshold, or synaptic strength.
- **No neuropeptides**, no NPF, no volume transmission, no internal/hunger state.
- **No plasticity.** Weights are fixed. Nothing learns inside the brain model - all learning in your RL setup has to live outside it, in the readout or the input encoding.
- **No gap junctions.** Not detectable in the EM volume, so ignored entirely. Some real circuits (notably the Giant Fiber escape pathway) are gap-junction dependent and will be wrong.
- **No morphology, no dendritic compartments, no receptor dynamics.** Every neuron is a single point with identical parameters - the same 20 ms time constant, the same threshold, the same refractory period, whether it is a photoreceptor or a Kenyon cell.
- **One sign per neuron.** Dale's law imposed at the neuron level via a majority vote over synapses. Co-transmitting neurons are misrepresented.
- **Glutamate assumed inhibitory.** Real Drosophila glutamate can be either. Flipping this assumption breaks the aversive-taste result and 16x's the optogenetic false-positive rate.
- **No non-spiking neurons.** A substantial fraction of the real fly brain (many local interneurons, all photoreceptors) is graded. All are forced to spike here.
- **Accuracy is bounded by the upstream predictions.** Synapse detection (Buhmann et al.) and NT prediction (Eckstein et al.) both have error rates that propagate straight into the weights.
- **Weights carry no distance or compartment information.** A 30-synapse edge onto a distal dendrite and onto the soma are identical.
- **Uniform 1.8 ms delay everywhere.** Real conduction delays vary with axon length; a brain-spanning projection and a local microcircuit get the same delay.
- Code-level: the `w = 0` in the reset statement is a no-op; `Excitatory` is pre-baked in the data so you cannot re-derive signs; `Presynaptic_Index` depends on CSV row order so cached matrices are silently invalidated by any reordering.

---

## Open questions for spec

- **Connectome dataset mismatch.** Every number in this document is FlyWire (female, brain-only, v630 or v783). MaleCNS v1.0 is a different specimen, a different reconstruction, includes the VNC, and has its own neurotransmitter prediction pipeline. Which MaleCNS tables carry the NT prediction, and does it expose per-synapse predictions plus a cleft-score-equivalent so the Shiu majority-vote rule can be applied faithfully? If it only ships a per-neuron NT label, the signing rule has to change.
- **The `>= 5` threshold is our choice, not the paper's.** It removes 82% of edges and 37% of synaptic mass. Are we willing to recalibrate $W_{\text{syn}}$ upward to compensate, and if so against what target? There is no MaleCNS equivalent of the "sugar GRN @ 100 Hz gives 80% of maximal MN9" anchor.
- **What is the validation target for the port?** The only exact ground truth available is FlyWire v630 (MN9 at 93.27 Hz for 200 Hz sugar drive, etc.). Do we build a FlyWire-v630 path purely as a correctness test for the JAX kernels, then swap in MaleCNS? That seems worth the cost but doubles the data ingestion work.
- **Does brain state persist across Craftax actions?** Carrying `v`, `g` and the delay buffer across the action boundary gives the agent short-term memory (~20 ms of it, set by $T_{mbr}$). Resetting each action makes the brain a pure stateless function of the current observation. These are very different agents. Given the zero-baseline assumption, persistent state decays to nothing within ~100 ms anyway.
- **Steps per action and `dt`.** 250 steps at `dt = 0.2 ms` is my recommendation but it is a guess. Needs a throughput measurement on the actual 8 GB card before the env loop is designed around it.
- **How is observation encoded into drive rates?** Craftax gives a symbolic/pixel observation; the model's only input channel is "Poisson rate on a chosen neuron set". Which MaleCNS sensory neurons, and what is the mapping from observation features to Hz? The paper's range is 10-260 Hz.
- **How is action decoded from spike counts?** desktop-fly's DN mapping (DNp09 forward, MDN backward, DNa01/02 steering, DNg11 grooming) is the obvious template but it is unvalidated, and Craftax's action space (move, place, craft, attack) does not map onto fly locomotion. Do we use identified DNs at all, or just train a linear readout on an arbitrary output population?
- **Licensing.** `external/fly-brain/` is GPL-2.0-or-later except `code/paper-phil-drosophila/` which is MIT. What license do we want for this project, and does that rule out reading `run_pytorch.py` while writing our step function?
- **Do we need silencing at all?** It is the paper's main causal tool but has no obvious role in an RL loop. If we skip it we save nothing (the mask is free), so probably keep it for interpretability experiments - but confirm before building tooling around it.
