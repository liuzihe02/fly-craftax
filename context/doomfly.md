# doomfly (nftechie/doomfly) — reference notes

Source: `external/doomfly/` (local clone), cross-checked against
`https://github.com/nftechie/doomfly` via the GitHub API (no issues, no
releases, no homepage/blog link found on the repo). All line numbers below are
from the local clone as read on 2026-09-09.

MaleCNS v1.0 connectome driving live ViZDoom via a LIF spiking model. Status
per the repo's own README: **"live experimental training, not demonstrated
learned survival"** — the project never got past its own validation gates.

## What doomfly actually is

- A whole-connectome (166,700 neuron) LIF simulator wired to real ViZDoom
  pixels on one side and to Doom's turn/move/fire buttons on the other.
- A fixed (hand-picked, not trained) readout: two descending-neuron types map
  to controls.
- An optional dopamine-gated plasticity rule on ~4,184 KC to MBON11 edges,
  added late (v5/v6) and never validated.
- Extremely heavy internal auditing culture: every claim in the README is
  cross-checked by a dedicated review doc, and the authors explicitly refuse
  to claim "learning" or "fly vision" anywhere.

## Repo layout

| Path | Contents |
|---|---|
| `doom/` | Core: MaleCNS importer (`connectome.py`), dataset registry (`datasets.json`), graph prep (`prepare.py`), reference Python LIF kernel (`engine.py`), native C++ kernel (`kernel.cpp`, built by `build_kernel.py`, loaded by `native.py`), ViZDoom adapter (`game.py`), server/broadcaster (`server.py`, `broadcast.py`), audits (`audit_data.py`, `audit_retina.py`, `audit_remote.py`, `audit_experiments.py`), BCI validation (`validate.py`, `validate_bci.py`), training loop (`training.py`, `training_checkpoint.py`), transmitter-sign policy (`transmitters.py`), reward stub (`reward.py`), combat arena WAD/scenario files. |
| `doom_learning/` … `doom_learning_v6/` | Six successive, mostly-failed learning candidates (see Section 7). Each is a near-complete fork of `doom/` with its own `brain.py`/`kernel.cpp`/`visual.py`/`conditioning.py`/`rule.py`. |
| `doom-ui/` | Next.js spectator website (live telemetry, "how it works", learning/methods pages). Not neuroscience-relevant. |
| `tests/` | pytest suite: numerical (Brian2 cross-checks), game, broadcast, checkpoint, learning-candidate tests per version. |
| `docs/` | The actual science: neuroscience review, learning review, iteration log, live-training protocol, performance review, prior-art review, one-hour QA report, combat-arena notes. This is where all the negative results live. |
| `data-provenance/`, `outputs/` | Locked source hashes (`source.lock.json`) and normalized-import report (`report.json`) for MaleCNS v1.0; machine-readable audit/experiment outputs. |
| `research/huang-2024/` | Extracted Figure 1/2 summary statistics (spontaneous DAN/MBON firing rates, shock response) from Huang, Luo et al. 2024, used to calibrate background currents in v5/v6. |
| `deploy/doomfly/` | Dockerfile, compose, native-observer patch for the public broadcast host. |
| `licenses/` | Third-party license texts (CC-BY-4.0 for MaleCNS, MIT for the Shiu model, ViZDoom MIT, Freedoom BSD-3). |

## MaleCNS importer

### Files downloaded and checksums

`doom/datasets.json` (lines 1-25) is the registry; only one dataset,
`malecns_v1`:

```json
"files": {
  "annotations.feather": ".../body-annotations-male-cns-v1.0-minconf-0.5.feather",
  "neurotransmitters.feather": ".../body-neurotransmitters-male-cns-v1.0.feather",
  "edges.feather": ".../connectome-weights-male-cns-v1.0-minconf-0.5.feather"
}
```

- All three files are hosted at `storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`.
- SHA-256 + byte size are locked in `data-provenance/malecns_v1/source.lock.json`
  (edges.feather alone is ~1.05 GB, sha256 `e35da78…`).
- `doom/connectome.py` `import_graph()` (lines 98-115) re-hashes every local
  file on each run and raises if a hash drifts from the lock file — it never
  silently re-imports a changed source.
- The upstream files already carry a **published pre/post synapse confidence
  threshold of 0.5** baked into the filenames (`minconf-0.5`) — that filtering
  happened at Janelia, not in doomfly's own code.

### Node (neuron) filtering: brain vs VNC, and quality filter

`doom/connectome.py` `normalize_nodes()` (lines 60-95):

```python
# doom/connectome.py:64-70
if dataset_id == "malecns_v1":
    source = exact_ids(frame.bodyId)
    retain = frame.superclass.notna() & frame.superclass.astype(str).ne("")
    quality = frame.statusLabel.astype(object).fillna("unknown")
    superclasses, types = frame.superclass, frame.type
    nonneural = frame.status.eq("Glia")
    retain = retain & ~nonneural
```

- **No brain-vs-VNC split.** `coverage: "brain_and_ventral_nerve_cord"` in
  `datasets.json` — doomfly keeps the entire CNS (brain + VNC) as one graph,
  never crops to brain-only. Confirmed by AGENTS.md: *"Retain every released
  connection between the retained neuronal entries. Do not crop circuits..."*
- Node inclusion rule = "has a non-empty `superclass` annotation AND is not
  explicitly labeled Glia." That's it — no restriction to `Traced` status, no
  restriction to typed cells; uncertain `_tbc` superclasses are kept.
- Result (from `data-provenance/malecns_v1/normalized/report.json`):
  - 211,577 raw annotation rows
  - 166,700 retained as neuron candidates
  - 33,013 excluded as "unresolved_object" (no superclass)
  - 11,864 excluded as "non_neuronal" (Glia)
  - 217 of the retained neurons end up with zero in/out edges ("isolated")
  - Superclass breakdown: `ol_intrinsic` 89,403 (optic lobe intrinsic — more
    than half the graph), `cb_intrinsic` 32,164, `vnc_intrinsic` 13,161,
    `visual_projection` 9,201, `descending_neuron` only 1,314, `vnc_motor` 708.

### Edge filtering and synapse-count threshold

`doom/datasets.json:12`: `"edge_policy": "All released edges between retained
entries; no additional weight threshold or removal of self-connections."`

`doom/connectome.py` `index_edges()` (lines 34-49) only rejects an edge row if
an endpoint isn't in the retained node set, or if the synapse count fails a
basic sanity check (positive, integral, fits uint32) — there is **no minimum
synapse-count / synaptic-weight threshold** beyond Janelia's upstream 0.5
confidence cutoff:

```python
# doom/connectome.py:42-49
if (not np.all(np.isfinite(counts)) or np.any(counts < 1)
        or np.any(counts != np.floor(counts)) or np.any(counts > 2**32 - 1)):
    raise ValueError("Synapse counts must be positive uint32-compatible integers.")
i, j = np.searchsorted(ids, pre), np.searchsorted(ids, post)
keep = (i < len(ids)) & (j < len(ids))
keep &= ids[np.minimum(i, len(ids) - 1)] == pre
keep &= ids[np.minimum(j, len(ids) - 1)] == post
```

`report.json.additional_edge_strength_threshold` is explicitly `null`, and
`upstream_autapses_excluded: false` — self-edges are kept (101 retained).

Resulting edge counts (same report):

| Quantity | Value |
|---|---:|
| Raw edge rows examined | 151,856,684 |
| Retained edge rows | 25,582,938 |
| Excluded edge rows (endpoint not retained) | 126,273,746 |
| Raw synaptic contacts | 311,833,243 |
| Retained synaptic contacts | 124,177,617 |
| Single-contact ("weight-one") edges retained | 10,299,701 |
| Self-edges retained | 101 |

### Neurotransmitter sign assignment

`doom/transmitters.py` (full file, 21 lines):

```python
# doom/transmitters.py:5-21
def transmitter_signs(transmitters, ambiguous_sign=1):
    """Declared coarse fast-transmission assumption; never deletes unknown edges.

    ACh +; GABA, glutamate, histamine -. A co-transmitter combination with only
    one fast sign uses that sign; conflicting, missing and modulator-only cells
    use the explicit sensitivity parameter. This is NOT receptor physiology.
    """
    if ambiguous_sign not in (-1, 1):
        raise ValueError("Ambiguous edges must remain active with sign +1 or -1.")
    signs, uncertain = [], []
    for value in transmitters:
        tokens = set(str(value).lower().split(','))
        fast = ({1} if 'acetylcholine' in tokens else set()) | ({-1} if tokens & {'gaba', 'glutamate', 'histamine'} else set())
        ambiguous = len(fast) != 1
        signs.append(ambiguous_sign if ambiguous else next(iter(fast)))
        uncertain.append(ambiguous)
    return np.asarray(signs, dtype=np.int8), np.asarray(uncertain, dtype=bool)
```

- Sign is a **per-presynaptic-neuron property**, applied to all of that
  neuron's outgoing edges uniformly (no per-target receptor logic).
- ACh -> +1, GABA/glutamate/histamine -> -1. Dopamine, octopamine, serotonin,
  "unclear", and missing all fall through to `ambiguous_sign`, default **+1**
  (treated as excitatory by default — flagged explicitly as "NOT receptor
  physiology").
- 3,718 neurons end up with an uncertain/default sign (per the neuroscience
  review). Neurotransmitter counts from the import report: acetylcholine
  103,720; glutamate 29,302; gaba 22,069; histamine 7,891; unclear 2,999;
  missing 178; dopamine 392; octopamine 101; serotonin 48.
- Final edge weight, in `doom/prepare.py:23`:
  `weight = synapse_count * sign(presynaptic_neuron) * 0.275` (mV-equivalent
  per synaptic contact — a single global gain constant borrowed from the
  Shiu et al. reference model, not fit to this connectome).

## Retina mapping

There is no literature "ring retina" term used anywhere in doomfly (grepped,
zero hits) — the geometry is a **flat overlapping-viewport hex-grid
embedding**, not a ring/annulus. Worth flagging as a naming mismatch with
whatever "ring retina" concept prompted this research task.

### Baseline: R1-R6 only, luminance-only (`doom/prepare.py`)

Column (retinotopic position) is inferred, not measured, from synaptic
contacts onto hex-coordinate-annotated lamina cells:

```python
# doom/prepare.py:27-51
receptor = a.type.eq('R1-R6').to_numpy()
anchors = a.type.isin(['L1','L2','L3']).to_numpy() & a.assignedOlHex1.notna().to_numpy()
selected = receptor[pre] & anchors[post]
# Infer column from the distribution of ALL R1-R6→L1/L2/L3 contact counts.
cols = {}
for i, j, w in zip(pre[selected], post[selected], count[selected]):
    key = (float(a.assignedOlHex1.iloc[j]), float(a.assignedOlHex2.iloc[j]))
    cols.setdefault(int(i), {}); cols[int(i)][key] = cols[int(i)].get(key, 0) + int(w)
indices = []; xy = []; confidence = []; hexes = []
for i, counts in sorted(cols.items()):
    h = max(counts, key=counts.get); total = sum(counts.values())
    indices.append(i); hexes.append(h); confidence.append(counts[h] / total)
    xy.append((h[0] - .5*h[1], np.sqrt(3)/2*h[1]))   # axial hex -> Cartesian
uv = np.empty_like(xy)
sides = a.rootSide.to_numpy()[indices]
for side in ['L', 'R']:
    mask = sides == side; z = xy[mask]; z = (z - z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0))
    uv[mask, 0] = (.60*z[:, 0] if side == 'L' else .40 + .60*(1 - z[:, 0]))
    uv[mask, 1] = 1 - z[:, 1]
```

- Each R1-R6 photoreceptor "votes" for a hex column via the total synaptic
  weight of its contacts onto hex-annotated L1/L2/L3 lamina cells; it's
  assigned the modal (highest-weight) column. `confidence` = fraction of that
  neuron's contact weight going to the winning column (median ~1.0, but 22-ish
  cells fall below 0.8 confidence in later audits).
- Axial hex coords `(h1,h2)` -> Cartesian `(h1 - 0.5*h2, sqrt(3)/2 * h2)`, then
  each eye (`rootSide` L/R) is independently min-max normalized to `[0,1]`.
- **Left/right split is not "R1-R6 numbers split into two halves of one
  panorama"** — both eyes are placed into **overlapping** UV viewports: left
  eye occupies U in `[0, 0.6]`, right eye occupies U in `[0.4, 1.0]` (mirrored),
  V full-height for both. This is explicitly called out as an "experimental
  overlapping viewport projection, not calibrated retinal angles" (manifest
  string, `prepare.py:59`).
- Counts: baseline is **3,335 mapped R1-R6 receptors** (42 more R1-R6 cells
  exist in the graph but get no pixel — no anchor contact found), split
  **1,107 left / 2,228 right** (asymmetric — noted as a potential bias in
  `docs/doom-neuroscience-review.md`).
- Pixel-to-current: `doom/game.py:92-100` (`retinal_samples`) bilinearly
  samples the raw ViZDoom RGB screen buffer at each receptor's UV, converts
  sRGB to linear luminance via the standard Rec.709/sRGB formula
  (`0.2126 R + 0.7152 G + 0.0722 B` after gamma-decode) — **brightness only,
  no color**, at this baseline stage.
- Current injection, `doom/engine.py:84-92` / `doom/kernel.cpp` drive array:

```python
# doom/engine.py:84-93
alpha = 1 - math.exp(-steps*self.dt/10)          # first-order low-pass, ~10 ms tau
self.luminance += alpha*(np.clip(luminance,0,1) - self.luminance)
self.drive.fill(0)
self.drive[self.lamina] = lamina_bias            # tonic 12 mV-equivalent, ALWAYS on
self.drive[self.retina] = 30*self.luminance/(.02+self.luminance)   # saturating drive
```

  Filtered luminance -> a saturating (Michaelis-Menten-shaped) current
  `30*L/(0.02+L)` mV-equivalent injected as constant drive onto the
  photoreceptor node. Lamina cells (L1/L2/L3/L5) get a constant 12 mV tonic
  bias current regardless of image content — this tonic term turns out to be
  load-bearing for the whole downstream "black screen still produces action"
  failure (Section 7).

### Color extension: R8p/R8y (v2 through v6, `doom_learning_v6/visual.py`)

```python
# doom_learning_v6/visual.py:70-89 (rgb_step)
values = frame[y, x, self.r8_channel].astype(np.float32)/255   # pick B or G channel per cell
values = np.where(values<=.04045, values/12.92, ((values+.055)/1.055)**2.4)   # sRGB->linear
self.r8_light += (1-math.exp(-round(duration_ms/self.dt)*self.dt/10))*(values-self.r8_light)
pulses.append((self.r8, 30*self.r8_light/(.02+self.r8_light)))
return self.step(retinal_samples(frame, self.uv), duration_ms, stimulation=pulses, **kwargs)
```

- Same column-inference technique as R1-R6, but the "anchor" is *any*
  hex-annotated downstream target (not restricted to L1/L2/L3), and the R8
  eye-viewport transform is re-fit onto the *existing* R1-R6 eye bounding
  box so the color inputs land in the same screen region as the R1-R6
  brightness inputs (`doom_learning_v6/visual.py:29-41`).
- Color channel is a crude per-cell-type proxy: `R8p -> linear-sRGB Blue`,
  `R8y -> linear-sRGB Green` (`r8_channel = where(type=='R8p', 2, 1)`,
  i.e. numpy RGB channel index). No red channel, no UV, no R7.
- Counts (from `doom_learning_iteration_log.md`): 330 R8p + 481 R8y = **811
  color inputs**, matching the README's headline "3,335 + 811" figure.
- R8->aMe12 edges get their sign force-corrected to positive
  (`self.weight[e] = np.abs(self.weight[e])`) based on a specific citation
  (Xiao et al. 2023) — the only place doomfly manually overrides its own
  generic NT-sign rule for a target-specific reason.

## Internal-state inputs

There is **no hunger/satiety/general internal-state model**. Two narrow,
explicitly-labeled artificial inputs exist, both gated by *game* state (not
biological drive):

1. **Aversive/damage signal -> PPL101 (dopaminergic) cells.** Nonfatal health
   loss schedules a 200 ms, +4 mV-equivalent current pulse into exactly two
   identified PPL101 neurons (IDs 11327, 11900):
   ```python
   # doom_learning_v6/survival.py:54-55
   damage = max(0, previous['health'] - state['health']); previous = state['health']
   if schedule is None and damage > 0: until = b.cursor + 2000   # 2000 steps @ dt=0.1ms = 200ms
   ```
   and delivered as `stimulation=(b.circuit['dan'], 4.)` in the same file
   (line 50). This is the closest thing to a "pain"/internal-state signal.
2. **Reward/"sugar" signal -> LB3c cells.** `doom/engine.py:93`
   (`if sugar: self.drive[self.sugar]=30`) and the manifest's `sugar` array is
   built from `cell_type == 'LB3c'` (`doom/prepare.py:67`) — 23 cells. A
   positive game-score delta can trigger a 200 ms, 30 mV pulse into these
   cells. **Disabled in the public baseline broadcast.** No learning is
   attached to this pathway either — it's pure stimulation.

No hunger, no proprioception, no internal clock, no other body-state variable
enters the network anywhere in the codebase.

## DN readout

Two parallel, both **hand-selected, never trained**, decoder modes in
`doom/engine.py` `NeuralControls.decode()` (lines 102-127):

- **`biological` mode** (role-matched cells): `DNa02` R-L rate -> turn,
  `DNp09` minus `MDN` rate -> forward, any `MN9` spike -> attack. Per the
  status doc, this mode produced **zero actions** under the tested visual
  conditions and was abandoned for the public feed.
- **`bci` mode** (the one actually used publicly): four specific
  cells, picked *because* they were visually responsive during manual
  calibration, with fixed IDs printed in `docs/doom-status.md`:

  | Cell | Source ID | Role |
  |---|---:|---|
  | DNp20 right | 10059 | turn |
  | DNp20 left | 10162 | turn |
  | DNpe017 left | 10527 | forward + attack |
  | DNpe017 right | 555871 | forward + attack |

  ```python
  # doom/engine.py:120-125
  if self.mode == 'bci':
      turn = float(np.clip((rate('DNp20','R') - rate('DNp20','L'))*.12, -6, 6))
      forward = float(np.clip(rate('DNpe017')*.4, 0, 20))
      attack = any(counts[r['index']] > 0 for r in self.readouts if r['type'] == 'DNpe017')
  ```
- Spike counts are converted to rate via a 100 ms exponential filter
  (`self.rates = self.rates*exp(-dt/0.1) + raw*(1-exp(-dt/0.1))`,
  `engine.py:111`), then the above **hand-tuned linear gains + clip** produce
  turn/forward, and **any nonzero DNpe017 spike in the interval presses fire**
  (not a threshold on rate, a raw "any spike" test).
- Explicitly no argmax over discrete actions, no softmax, no trained linear
  probe, no RL policy head of any kind — this is manually-selected cells with
  manually-chosen scalar gains, described repeatedly in the docs as "an
  engineered joystick mapping, not a biological interpretation."

## Brain dynamics

- **Model**: single-compartment LIF, identical equations everywhere in the
  codebase (`doom/engine.py:1-7`, `doom/kernel.cpp:1-3`):
  ```
  dv/dt = (-52mV - v + drive + g) / 20ms
  dg/dt = -g / 5ms
  spike when v > -45mV; reset v=-52mV, g=0
  refractory = 2.2 ms; synaptic delay = 1.8 ms; dt = 0.1 ms (hard-enforced, no other dt supported)
  edge increment (on arrival) = contact_count * 0.275 mV * presynaptic_sign
  ```
  Parameters are explicitly borrowed from the Shiu et al. 2024 reference LIF
  model (constants, not refit), acknowledged in `docs/doom-neuroscience-review.md`
  section 3 as "not a direct replication... different connectome, different
  input, different task."
- **Substeps per game tic**: ViZDoom runs at 35 Hz (~28.57 ms/tic); each tic
  advances the LIF network **285 or 286 substeps** of 0.1 ms
  (`doom-neuroscience-review.md` section 7: "advances the neural model by 285
  or 286 substeps... advances the game by exactly one tic").
- **Libraries / implementations, three separate ones**:
  1. `doom/engine.py` `Brain` — pure-Python reference kernel, JIT'd with
     **numba `@njit`** (not torch, not JAX).
  2. `doom/native.py` `NativeBrain` / `doom/kernel.cpp` — hand-written C++
     kernel, compiled at build time with `clang++ -O3 -std=c++17 -shared
     -fPIC` (`doom/build_kernel.py:12`), loaded via `ctypes`. This is the
     production path (used by the public server and all `doom_learning_v*`
     candidates). It's an event-driven / lazy-update kernel (only evolves
     "active" neurons, exact analytic subthreshold solution between events)
     rather than a dense per-step update over all 166,700 neurons.
  3. **Brian2 2.5.1** is used only as an **independent numerical oracle** for
     validation (`requirements-neural.txt`), never as the runtime — a
     dedicated isolated venv exists just to cross-check the custom kernel
     against Brian2's semantics (this is how the refractory/synaptic-write bug
     below was caught).
- **No GPU anywhere.** Purely CPU: numba JIT or hand-written serial C++ SIMD-ish
  kernel, single-threaded. Nothing in the repo touches CUDA/Metal/OpenCL.
- **Throughput (all CPU, Apple M1 Pro 10-core / 16 GB, shared with other load)**:
  - Public baseline server: **7.28 game tics/sec (0.208x real time)**, publishing
    5.17 frames/sec (`docs/doom-performance-review.md`); a later 293 s sample:
    6.84 tics/sec, 4.99 fps.
  - Native-kernel step cost: median **75.1 ms** wall-clock to simulate one
    28.6 ms game interval (mean 121.2 ms, max 1158.6 ms) — i.e. the kernel
    alone is >2.5x slower than real time at the median, with heavy-tailed
    spikes to 40x.
  - After removing BLAS thread oversubscription
    (`OPENBLAS_NUM_THREADS=1`): a 20.23 s sample reached **0.662x real time**
    (13.40 neural seconds simulated), 8.10 published fps — best throughput
    recorded anywhere in the repo, still well under real time.
  - `doom_learning` (v1, no plasticity) pilot: 260.91 neural seconds in
    1240.76 wall seconds = **0.210x**.
  - `doom_learning_v6` survival pilot: 80.23 brain-seconds in 490.74 wall
    seconds = **0.163x**; neural integration itself consumed 91.9% of wall
    time (i.e. ViZDoom stepping/rendering is not the bottleneck, the LIF
    kernel is).
  - A serial-vs-parallel scheduling micro-benchmark found only ~1.12x
    aggregate speedup from running 2 episodes in parallel processes on the
    shared host (confounded by setup/host load, explicitly caveated as
    unreliable).
  - **Never reached real time (1x), let alone 30/60 FPS**, at any point in the
    project's history. The performance-review doc's own "path to 30 FPS"
    section says roughly a 4x further improvement over the *already-optimized*
    7.28 tics/sec would be needed just to hit real-time-equivalent 30 distinct
    states/sec.

## Documented failures and lessons (the important part)

This project's docs/ directory is essentially a running lab notebook of
negative results. Organized by root cause.

### 1. No plasticity in the public/baseline model at all

The headline baseline (`doom/`) has **zero synaptic plasticity**. Every
"kill" or "survival" number produced by the baseline is from a completely
fixed, untrained network. Quote, `docs/doom-neuroscience-review.md`:

> "No synaptic plasticity, weight updates, or validated memory mechanism
> exists." ... "Weights remained byte-identical through the new tests.
> Persistent membrane state across episodes is dynamics, not evidence of
> learned skill."

### 2. A silent numerical bug inflated activity for an unknown period

> "Both local solvers froze `g` during refractoriness but still added
> incoming weights. Brian2's `unless refractory` makes the variable read-only,
> including synaptic writes. A strong autapse and convergent excitation/
> inhibition exposed 39 mismatched neuron/time bins in a 4-cell, 90 ms test
> before the fix." (`doom-neuroscience-review.md` section 3)

I.e. refractory neurons were still silently accumulating synaptic current
that got applied the instant refractoriness ended, an unintended
"memory"/integration effect. Caught only by an independent Brian2
cross-check, not by inspection. Relevant for anyone hand-rolling a custom LIF
kernel: get an independent oracle early.

### 3. Actions fire from tonic drive alone — vision may not matter

This is the most damning result in the repo. The lamina gets a **constant 12
mV tonic current regardless of pixel content** (`doom/engine.py:91`), and
that alone is enough to drive the DN readout:

> "Black visual input still permits controls to fire because lamina current
> supplies a nonvisual drive. Removing that current silences the selected
> controls in the fixed-frame test, even with the original image present."
> (`doom-neuroscience-review.md` section 8)

And the closed-loop kill counts actively *favor* no vision:

| Seed | Live vision kills | Black input kills | Clamped-control kills |
|---|---:|---:|---:|
| 41027 | 1 | 3 | 0 |
| 41028 | 1 | 3 | 0 |
| 41029 | 1 | 2 | 0 |

> "Black input outperformed live vision in these short comparisons... the
> experiment provides no evidence of a visual-performance advantage."

This is a general warning: if any input channel carries a constant/tonic
bias term large enough to drive downstream spiking on its own, closed-loop
behavioral metrics (kills, survival time) cannot distinguish "the network is
using the sensory signal" from "the network is running on autopilot." Ablation
(black-frame, disconnect-all-edges) controls are mandatory, not optional.

### 4. The visual pathway never produces distinguishable, sparse memory-relevant activity

Across nearly every candidate version, the same finding recurs: KC (Kenyon
cell, the fly's sparse-coding/memory layer) populations are either **totally
silent** or **saturate to broad, non-selective activity** — never the sparse,
stimulus-selective code needed for associative learning.

- v1 baseline stimulus assay:
  > "all 6,865 T4 cells, 6,720 T5 cells and 4,064 Kenyon cells produced zero
  > spikes. All 206 annotated KCg-d cells were silent." (`doom-learning-review.md`)
- v2 (adds R8 color + separated modulatory channels): blue image -> 283
  spikes in 14/206 KCg-d cells over 1 second; green produced **zero**
  activation of that same memory population; opposite-side (left/right blue)
  half-fields recruited only 10 and 5 cells respectively.
- v4 (adds KC spike-frequency adaptation specifically to fix over-activity):
  white-stimulus KC spike count dropped from 36,169 to 7,860 but *still*
  recruited 1,475 of 4,064 KCs (mostly generic gamma-main/alpha-beta types,
  only 3 of the visual-input-specific KCg-d cells fired) — adaptation reduced
  volume but not selectivity.
- v6 (adaptation + centered memory rule combined): **no KC spikes during dark
  baseline**, but after any cue (left blue / right blue / white) the network
  settles into a *persistently* elevated state that never recovers:
  > "final recovery KC rates are respectively 8,950, 8,935 and 8,976 spikes/s
  > across the population" (3 seconds after cue offset) "Corresponding PPL
  > rates are 106/105, 108/95 and 108/105 Hz, versus approximately 22/18 Hz
  > before the cue." (`doom-learning-iteration-log.md`)

  and separately, from `doom-learning-review.md`, an even more extreme
  instance of the same pathology in v5:
  > "after left-cue presentation, the frozen model produced 276,709 KC spikes
  > in the final dark second, with PPL rates around 313 Hz. This persistent
  > recurrent state cannot be treated as a validated representation or a
  > physiological reinforcement response."

  This reads as a **runaway recurrent excitation / no working inhibitory
  stabilization** failure mode: once activity in the loosely-parameterized
  (uniform gain, uniform time constants, blanket-sign NT) network gets kicked
  above some level, it doesn't settle back to baseline within seconds.

### 5. Conditioning assays fail their own no-imposed-US control

Every version's associative-learning test uses the same predeclared gate: a
paired (cue + reinforcement) condition should show a learning effect that a
matched *unpaired*/no-US control does not. It never passes:

- v1/v2/v3 direct-KC-stimulation conditioning: "paired MBON output increased,
  and no-imposed-US and backward conditions also changed a small number of
  synapses" — the effect isn't cue-specific.
- v4 (counterbalanced, cleanest test): 
  > "Both forward-paired and no-imposed-US training suppressed both cue
  > responses to zero. Frozen responses were identical before and after
  > training. The predeclared cue-selectivity gate failed."
- v5 (centered rule, most detailed):
  > "With the left cue paired, its MBON response fell by 93.79%, but the
  > no-imposed-US control showed exactly the same output suppression. The
  > opposite cue fell by 4.88% in both. With the right cue paired, suppression
  > was greater for the *wrong* cue."

  The stated diagnosis: endogenous/recurrent dopaminergic (PPL101) activity
  fires on its own from the sensory drive, independent of the imposed
  "reinforcement" pulse, so the plasticity rule can't tell paired from
  unpaired trials — the "reward" signal is swamped by uncontrolled internal
  dynamics.

### 6. Even where weights changed, the effect was harmful, not adaptive

v6's actual ViZDoom survival pilot (12 episodes, blue-hazard-floor arena):

> "On held-out seed 61041, learning-on and shuffled both died at 3.657 s,
> while frozen reached the 8 s cap with 60 health. On seed 61042 all arms
> died at 3.657 s." ... "Erasing memory restored the entire frozen
> input/spike/action/health/position trace on seed 61041, including survival
> to the cap. Thus memory changes caused a behavioral difference in this
> model, but the observed difference was harmful."

I.e. plasticity was demonstrably causal (memory erasure fully reverted
behavior) but made survival strictly *worse* than the frozen, untrained
network. Also notable: the *shuffled-reinforcement-timing* control learned
the same harmful change as the real-timing arm, meaning the "learning" wasn't
even using correct temporal pairing — some other confound (again, likely the
same uncontrolled recurrent dynamics from #4/#5) was driving the weight
changes.

### 7. Performance never reached real time, on any measured configuration

Best-ever recorded speed was 0.662x real time, after removing a
BLAS-thread-oversubscription bug (9 spin-waiting helper threads while the
sim thread waited inside `cblas_sgemv64_` for a simple 3-channel color
transform — `docs/doom-performance-review.md`, item 2). Typical operating
speed for actual experiments was 0.16-0.21x. The kernel itself (event-driven,
lazy-update C++, single core) was the dominant cost, not ViZDoom or I/O. No
GPU port was ever attempted; the docs list it only as a "candidate, not a
measured speedup."

### 8. Author's own meta-lesson on claims discipline

Recurring language throughout every doc, worth internalizing as project
culture:

> "Changed weights and longer individual rounds do not establish learning."
> (README)

> "A defensible learning experiment requires identified sensory/reinforcement
> paths, compartment-specific plasticity supported by physiological data,
> explicit state and weight measurements, and frozen-decoder evaluation.
> Compare pretraining/post-training behavior with no reward, shuffled or
> yoked reward, plasticity disabled, and held-out game seeds. Demonstrate
> retention and loss of the effect under a relevant causal intervention
> before claiming learning." (`doom-neuroscience-review.md` section 6)

The project set this bar for itself early and then, by its own final
iteration-log entry, still had not cleared it: *"The announcement gate
remains NOT READY."*

## Reusable material (verbatim), with license notes

- **MaleCNS v1.0 data itself**: CC BY 4.0 (Janelia FlyEM / MaleCNS
  collaboration), see `licenses/CC-BY-4.0.txt` and `THIRD_PARTY.md`. Same
  dataset we'd use — no doomfly-specific rights involved, just standard
  attribution to the consortium and the paper (`doi.org/10.1016/j.cell.2026.08.015`).
- **doomfly's own Python/C++ code**: MIT (`LICENSE`, copyright "nftechie and
  DOOMFLY contributors"). Directly reusable with attribution:
  - `doom/connectome.py` — the checksum-locked importer pattern
    (`file_digest`, lock-file comparison, `exact_ids` avoiding float rounding
    of 64-bit body IDs) is a clean, small, reusable pattern regardless of
    downstream framework (JAX/etc.).
  - `doom/transmitters.py` — the ACh/GABA-glutamate-histamine sign convention
    is a reasonable, tiny starting point for a coarse E/I sign if we don't
    want to build full receptor-type logic immediately; but note it's
    explicitly flagged by the authors as "NOT receptor physiology," and
    dopamine/octopamine/serotonin default to *excitatory* which may be the
    wrong default for a from-scratch design.
  - `doom/prepare.py`'s modal-hex-column retinotopy inference (contact-weighted
    voting onto hex-annotated lamina cells) is a reusable *technique* for
    building any custom photoreceptor-to-screen mapping from MaleCNS/optic-lobe
    hex annotations, independent of Craftax specifics.
  - The LIF kernel equations/constants (`doom/engine.py`, `doom/kernel.cpp`)
    are themselves inherited from Shiu et al. 2024's model (**MIT, "copyright
    Philip Shiu and Nico Spiller"**, `licenses/Shiu-model-MIT.txt`) — if we
    reuse the specific numeric constants (dv/dt, tau, thresholds, 0.275 mV
    contact gain), that upstream attribution should be carried, not just
    doomfly's.
- **The Huang, Luo et al. 2024 calibration constants** (background firing
  rates used in v5/v6, `research/huang-2024/targets.json`) are CC BY 4.0 per
  the paper; doomfly's extraction script is MIT but the underlying numbers are
  the paper's.
- **Not reusable / explicitly not bundled**: ViZDoom itself, Freedoom assets
  (BSD-3, game art, not relevant to us), the doom-ui Next.js spectator site
  (irrelevant to a JAX/Craftax project).

## Lessons for our design

- Ablation controls are not optional: any constant/tonic bias current (or
  analogous "always-on" input) can make closed-loop behavior look
  input-driven when it is not. Build black/scrambled/disconnected-input
  controls into the pipeline from day one, not as an afterthought review.
- Get an independent numerical oracle (even a slow, small reference
  implementation) before trusting any hand-optimized custom kernel — doomfly's
  refractory-period bug shipped silently until a Brian2 cross-check caught it.
- Sparse, stimulus-selective activity in a memory-relevant population (their
  KCs; whatever the fly-Craftax equivalent input/association layer is) is not
  a given from anatomy alone — doomfly never got past broad
  saturation/silence in *any* of six candidate iterations. Expect to spend
  real effort calibrating background drive, adaptation, and inhibition before
  plasticity can even be meaningfully tested; do this with cheap, isolated
  stimulus assays before wiring up the full game loop, exactly as doomfly's
  later iterations tried to do (too late).
- A no-imposed-reinforcement control that shows the *same* effect as the
  paired condition means the plasticity signal is dominated by something
  else (endogenous/recurrent activity in their case). Always run that control
  before trusting any weight-change result.
- Memory-erasure / frozen-vs-plastic A/B is a cheap, effective causal test
  (doomfly used it well) — worth adopting even though their result showed
  harm rather than benefit.
- Real-time throughput on CPU-only, dense whole-connectome LIF simulation of
  ~166k neurons was never achieved (best 0.66x, typically 0.16-0.21x) even
  with hand-written C++ and event-driven lazy updates. JAX (our stack)
  should be evaluated for both step-time and steps-per-Craftax-frame budget
  early, with a target FPS in mind, rather than discovered as a bottleneck
  late like doomfly's performance-review doc did.
- A fixed, hand-selected DN-to-action readout (no training, no argmax over a
  learned head) is workable as a first pass but doomfly's own experience
  (biological-role mapping produced literally zero actions; had to fall back
  to an ad hoc "whichever DNs happen to respond" BCI mapping) suggests we
  should budget for readout calibration/selection as real work, not a
  one-line mapping.
- Publish/track negative results as a first-class artifact (their docs/
  directory) — the project's scientific credibility comes entirely from
  documenting what *didn't* work, which is exactly the kind of context this
  file is meant to preserve for us too.

## Open questions for spec

- Do we want a coarse per-neuron NT sign convention (ACh+/GABA-glutamate-histamine-)
  like doomfly, full receptor-type-resolved signs, or a learned/parameterized
  sign per edge? doomfly's default-excitatory fallback for
  dopamine/octopamine/serotonin/unclear (3,718 cells) is a real design smell
  worth deciding on deliberately rather than inheriting.
- Do we crop to brain-only (dropping the ~13,161 `vnc_intrinsic` + 708
  `vnc_motor` + other VNC-superclass neurons) or keep brain+VNC as doomfly
  did ("no cropping" was a hard project rule for them, driven by their
  "retain the whole reconstruction" ethos)? Craftax's action space is much
  smaller/more abstract than Doom's continuous turn/move, so the VNC's motor
  neurons may not map as usefully.
- Should our retina/sensory encoding avoid any constant tonic bias term
  entirely, given doomfly's #3 failure (actions firing from lamina tonic
  current alone, independent of pixels)? If we do want a baseline/resting
  drive for numerical stability, how do we structurally prevent it from being
  large enough to single-handedly drive the readout?
- Craftax observations aren't raw RGB frames pointed at a virtual eye the way
  ViZDoom's screen buffer is — what does "photoreceptor" input even mean for
  a symbolic/grid observation? doomfly's whole retina-mapping approach
  (UV coordinates on a rendered frame) may not transfer; we may need a
  fundamentally different sensory-encoding scheme (e.g., mapping grid cells
  directly to receptor indices) rather than reusing their bilinear-sample
  code.
- Is there a target throughput (steps/sec, or Craftax-frames/sec) we need to
  hit for the RL loop to be practical, and does that rule out a literal
  166k-neuron dense LIF simulation regardless of JAX's speed advantage over
  doomfly's CPU kernel? Worth back-of-envelope estimating before committing
  to full-connectome retention the way doomfly did.
- Do we want to attempt any plasticity at all given how uniformly it failed
  here (harmful when causal, uncontrolled when not), or start from a fixed
  connectome + trained-readout-only design (skip their hardest, least
  successful axis entirely for v1)?
