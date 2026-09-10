# flybrain-craftax

Hobby RL project. A real fruit-fly connectome drives an agent in Craftax-Classic. Goal is to watch known fly circuits activate during play, train using RL and see how it plays or learns

## Bio Background

- See [context/bio_background.md](context/bio_background.md), a distilled non-biologist explanation of the brain, the LIF model, and every circuit in the I/O mapping

## Spec

High level project implementation specs

### Brain

- Population: MaleCNS v1.0, all non-VNC neurons (`superclass` set and not `vnc_*`), ~146k
  - Includes photoreceptors, visual projection, DNs, ascending neurons
- Edges: traced-only edge file; synapse threshold is a config value, default >= 5 (~5M edges)
- Sign by presynaptic consensus NT
  - ACh +; GABA, glutamate, histamine -
  - Dopamine, serotonin, octopamine + (as Shiu); `unclear` falls back to cell-type prediction, else edge dropped
- LIF per Shiu et al. 2024, exact exponential update, dt configurable (default 0.1 ms)
  - w = sign x synapse count x 0.275 mV; recalibrate the constant if the threshold changes
  - Input = Poisson forced spikes on a cell-type set; output = spike counts over a window; silencing = spike mask
  - State (v, g, delay buffer) persists across actions, resets per episode
- JAX, sparse BCOO matvec, batched over envs, fits the 8 GB 4060; 200 brain steps (20 ms) per env action, fixed after the M1 benchmark
  - Can rent RunPod if not enough
- Validation: Brian2 oracle (Shiu's MIT code) on a MaleCNS subgraph, spike-level agreement; sugar-GRN to MN9 curve on the full brain
- FlyWire v783 replication is a later milestone
- Tooling: conda env for Python, deps pinned in pyproject.toml. Project license MIT; port from Shiu's MIT code, never the GPL harness

### Env

- Craftax-Classic-Pixels-v1, two wrappers
  - Egocentric actions: `noop, forward, backward, turn_left, turn_right, do, sleep`
    - Turn costs a step (noop, then overwrite facing). Backward moves opposite and restores facing
  - Reward: +1 per new survival achievement (eat_cow, collect_drink, wake_up, defeat_zombie, defeat_skeleton, collect_sapling, collect_wood) plus 0.1 x health delta; other achievements zero
    - eat_plant dropped, unreachable without place_plant
- food, drink, energy read from the state struct

### I/O mapping

Inputs
- Pixels: inventory bar cropped; radial retina centred on the agent, azimuth = angle, elevation = distance, left hemifield to left eye
  - ~800 points per eye on the real hex column grid, inferred from photoreceptor to L1-L3 synapses
  - Brightness sets R1-R6 rate; colour sets R8 rate (R8p blue, R8y green)
- Hunger to NPF (`NPFL1-I`, 2 cells); thirst to hygrosensory class (66); fatigue to ER5 (21)
  - rate = deficit / 9 x max Hz, max Hz configurable
- Lamina cells L1-L5 get a tonic bias current so light, via photoreceptor inhibition, reduces a resting rate (agent ruling 2026-09-10, pending owner approval: all photoreceptors are histaminergic and purely inhibitory, so without a lamina baseline vision cannot propagate in a zero-baseline model)

Outputs (fixed mapping, zero-shot)
- forward: DNp09; backward: MDN; turn: DNa02 + DNa01 left-minus-right; do: MN9; sleep: FB6/FB7 tangentials
- action = argmax of signals, noop when all below a floor
- Signals are z-scored against a random-policy baseline with a one-spike-per-window floor; the two turn signals keep a zero baseline because their mean is antisymmetric by construction and subtracting the measured 4 Hz on the 25 Hz spike lattice inverted the standing bias (agent ruling 2026-09-10, pending owner approval; `flycraftax/readout.py` load_readout)
- Ablation controls built in from day one: black frame, shuffled weights, disconnected inputs

Trained alternative: brain frozen, linear readout over 1,314 DN rates to 7 logits, PPO

### Viewer

All are panels

- Craftax visualization
  - Want to see fly as sprite and any other realistic simulations
- Real-time interactive visualization using the existing libraries to see the activations
- Other indicators of the fly brain state or anything? TBD

### Milestones

- [x] M1 loader + JAX kernel + Brian2 validation, plan: `docs/plans/2026-09-09-m1-brain.md`
  - [x] scaffold, pinned env, data download
  - [x] connectome loader with cache
  - [x] Brian2 oracle
  - [x] JAX LIF kernel, spike-for-spike match on toy graphs
  - [x] oracle match on a 2000-neuron MaleCNS subgraph
  - [x] full-brain throughput benchmark on the 4090
    - 4090: 3,817 brain steps/s at batch 1, 152 steps/s at batch 32, under 1 GB. Sparse matvec is 90% of the step and gather-bound, so batching scales poorly
  - [x] MN9 smoke check with labellar GRN drive
    - 165 labellar GRNs at 100 Hz for 500 ms: MN9_L fires 2-10 Hz over seeds, MN9_R never (net-inhibitory in-edges), ~9.5k active neurons vs Shiu's 404. Rate does not rise with drive and drops above 100 Hz
    - threshold 2 fires MN9 less, so default 5 stays. Rate calibration is M3 work; full tables in `docs/plans/2026-09-09-m1-brain.md` results section
- [x] M2 Craftax wrappers + retina + state encoders, plan: `docs/plans/2026-09-10-m2-env.md`
  - [x] craftax installed and pinned, baseline wrappers copied, one env step
  - [x] egocentric action wrapper
  - [x] survival reward wrapper and batched env stack
  - [x] photoreceptor retinotopy from lamina and medulla synapses
    - 2,182 photoreceptors mapped onto 491 left and 585 right columns; 1,983 outer photoreceptors are untraced fragments absent from the traced-only edge file, so left-eye luminance covers only the outer half of elevations. The full 1.1 GB edge file would recover them (follow-up decision)
  - [x] radial retina sampler rotated by facing, visual check
  - [x] drive assembly: retina plus hunger, thirst, fatigue
  - [x] integration rollout through env, drive, brain, visual check
    - 40 random actions at batch 4: ~11k neurons spike but the output groups stay silent until the first thirst deficit at action 21, then MN9 and FB6/7 respond; DNp09 and MDN never fire. Interoception reaches the readout groups, the retina does not. M3 calibrates gain and synapse weight first
- [x] M3 zero-shot loop with ablation controls, plan: `docs/plans/2026-09-10-m3-zero-shot.md`
  - [x] tonic bias in the kernel, lamina resting drive
  - [x] DN readout: six signals, z-score floor
  - [x] closed-loop rollout as one scan with ablations and policies
  - [x] calibration sweep over lamina bias and synapse weight
    - pick 0.275/0.04, qualified (5 non-zero signals, active 0.053, black L1 above real L1); `w/mv: nonzero, active, L1 full/black` — 0.275/0.04: 5, 0.053, 17.7/23.8; 0.275/0.06: 5, 0.083, 45.3/52.9; 0.275/0.10: 6, 0.115, 91.4/97.8; 0.44/0.04: 4, 0.078, 15.9/22.9; 0.44/0.06: 4, 0.108, 42.0/52.0; 0.44/0.10: 4, 0.139, 86.3/96.4. Black beats real everywhere, so light does reduce lamina firing; no runaway (max active 0.139); DNp09 (`forward`) silent at every config but 0.275/0.10; every DN std sits at or under its floor, so the floor sets the gain. Ratified by the controller (first qualifying config). All six configs passed the rule, so the pick is grid order; the real justification is that light modulation of L1 falls with bias (26 / 14 / 7% at 0.04 / 0.06 / 0.10). The rule ranked 0.275/0.10, the only config where all six signals fire, below grid order.
  - [x] zero-shot evaluation against black, static, disconnected, shuffled, random
    - mean survival over 8 envs, 2,000-action cap: full 154.8, black 138.5, static 145.0, disconnected 168.0, shuffled 164.9, random 114.1. The per-env spread on `full` alone is 42 to 407, so no ablation separates from `full` on survival; the largest gap (random, -40.6) is under one standard error of the difference
    - total achievements over 8 envs: full 9, black 8, static 9, disconnected 0, shuffled 0, random 15 (mostly `wake_up`, plus `collect_wood`/`collect_sapling`). The dominant action of `full/readout` is `noop` at 0.68, then sleep 0.19, turn_left 0.12, do 0.01; DNp09 (`forward`) never fires. `black` and `static` match `full` action-for-action to within 0.01, so vision contributes nothing to behaviour; only `disconnected` (pure noop) and `shuffled` (turn_right 0.43, no achievements) change it
    - Caveats: survival and achievements are first-episode only (~155 of 2,000 actions) while the histogram and reward span all 2,000; conditions share seeds so the paired test applies (full vs random t = 0.97, conclusion unchanged); one shuffle permutation; `full`'s 9 achievements are 7 x `wake_up`, self-generated by its own sleep action; pure noop out-survives the wired fly (168 vs 155), so no adaptive control was shown; the 100:1 left-turn asymmetry rests on four single neurons and is non-visual; effective floors are 1/2/1/1/2/10 spikes per window for forward/backward/turn/turn/do/sleep; `forward` is exactly 0 in every unshuffled condition
- [x] M4 PPO linear readout, plan: `docs/plans/2026-09-10-m4-ppo.md`
  - [x] loop step factory with a parameterised linear policy
  - [x] PPO training function, fully jitted
  - [x] train, greedy evaluation against the M3 baselines
    - 150 updates, 76,800 env steps (8 envs x 64 steps), 58.6 min on the 4090, 550 episodes. Mean episode return 1.16 (first 10 updates) to 2.02 (last 10); episode length 132 to 130, i.e. the readout earns more reward per action without living longer. In-update entropy falls 1.89 to 0.99 of ln 7 = 1.95, so the policy is still far from deterministic. Loss stays finite throughout. The action mix moves from near-uniform (first 10: every action 0.12-0.16) to `do` 0.41 and `forward` 0.36 (last 10), through a `do`-heavy phase around update 60 and a `sleep`-heavy phase around update 90
    - greedy evaluation, 8 envs, 2,000-action cap, `PRNGKey(0)`, the same conditions as M3: mean survival ppo/greedy 117.6 (spread 45-244), full/readout 154.8, full/random 114.1; total achievements 21 (5 collect_wood, 7 collect_sapling, 3 collect_drink, 6 wake_up) versus 9 and 15; reward per action +0.0133 versus +0.0026 and +0.0104. The dominant action is `forward` 0.49 with `do` 0.46, against `noop` 0.68 for the zero-shot fly and a flat 0.14 for random. So PPO on the frozen brain's DN rates buys reward and achievements, not survival: it dies sooner than the zero-shot readout while collecting more. Caveats: one seed, one run; survival and achievements are first-episode only while the histogram spans all 2,000 actions; the training reward is the survival reward, and 76,800 steps is tiny for PPO
- [x] M5 viewer, plan: `docs/plans/2026-09-10-m5-viewer.md`
  - [x] fly sprites in the renderer, soma map, panel composition
  - [x] viewer script: frames, slider page, GIF
    - 300 actions of `full`/`readout` rendered eagerly: `outputs/viewer/frames/0000.png` .. `0299.png`, `outputs/viewer/index.html` (slider and play over the frames), `outputs/viewer.gif` (300 frames, 640x360, 10 fps). One colour scale for the whole run (95th percentile of non-zero window counts over the first 20 windows). Follow-up: a live 3-D view of the same activity with navis, instead of the 2-D soma projection over pre-rendered frames

## Known Limitations

- No neuromodulation: hunger, thirst, fatigue injected as spikes, not emergent
- No body, no VNC; DN activity read as discrete actions
- Zero baseline activity outside the lamina: the brain is silent unless driven, so inhibition is invisible on silent neurons. The lamina tonic bias has no Brian2 counterpart, so the oracle does not cover it
- Locomotion DN mapping is from optogenetics and hobby projects, not validated in the Shiu model
- Hunger is a 2-cell channel; thirst and fatigue use proxy cell types since water GRNs and dFB are unlabelled in MaleCNS
- MaleCNS has no sugar-specific GRN labels, so the MN9 validation drives all 165 labellar GRNs (sugar, bitter, water together)
- Fatigue via ER5 cannot propagate: ER5 is purely inhibitory into a zero-baseline network (follow-up decision)
- Lunge pathway dropped: pC1/aIPg is the female aggression circuit
- Two individuals exist in total (MaleCNS, FlyWire)
- Achievement score is a readout property, not a brain property

## Resources

- [craftax](https://craftaxenv.github.io/)
- [cns](https://male-cns.janelia.org/)

### Data

MaleCNS v1.0, CC-BY, public GCS bucket, no login. Directory:
`https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`
- `connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather` (508 MB) — edge list, synapse counts = weights, adds type columns; the 1.1 GB untraced variant is not needed
- `body-annotations-male-cns-v1.0-minconf-0.5.feather` (13 MB) — cell types, class, side
- `body-neurotransmitters-male-cns-v1.0.feather` (42 MB) — NT prediction per neuron -> edge sign
Docs: https://male-cns.janelia.org/download/
Loaders: https://github.com/sjcabs/fly_connectome_data_tutorial
FlyWire v783 (female) is the replication dataset; same pipeline, different loader.

### References

- **Env**
  - `MichaelTMatthews/Craftax` — JAX Crafter; use `Craftax-Classic-Pixels-v1`; achievement list; PPO baselines in `craftax_baselines`
- **Data loading**
  - `sjcabs/fly_connectome_data_tutorial` — MaleCNS / FlyWire edgelist + metadata loaders, class hierarchy
  - `flyconnectome/flywire_annotations` — cell-type labels for the FlyWire replication
  - `connectome-neuprint/neuprint-python` — query MaleCNS by cell type without bulk download
- **Brain dynamics**
  - `eonsystemspbc/fly-brain` — Shiu LIF model as code; thresholds, NT signing, activate/silence API; port step to JAX
  - `jax-ml/jax` — `jax.experimental.sparse.BCOO` for the sparse matvec
- **MaleCNS-specific glue**
  - `nftechie/doomfly` — checksummed MaleCNS importer, R1–R6/R8 retina mapping, DN readout, documented failures
- **Vision**
  - `NeLy-EPFL/flygym` — compound-eye ommatidia sampler (`vision/` module), ~800 inputs per eye
- **Visualization**
  - `navis-org/navis` + `navis-flybrains` — fetch meshes/skeletons, render spiking 3D brain for the dashboard
- **RL**
  - `luchris429/purejaxrl` — single-file PPO in JAX if the Craftax baselines are too tangled

### Other Prior Art

- Doomfly (MaleCNS + ViZDoom, failed to learn): https://github.com/nftechie/doomfly
- FlyGM (connectome as RL controller, MuJoCo): arXiv 2602.17997
- Digital Sphinx (worm brain walks a fly body — why the controls matter): eLife 2026
- Embodied FlyWire in NeuroMechFly: https://github.com/erojasoficial-byte/fly-brain
No connectome + Craftax/Crafter/Minigrid repo exists as of 2026-09-09.

## Agent Behaviour

> This design doc serves as the project spec - a crucial piece of living high level design document for how I see the project directions, plans, checklist etc.
> This doc should be mostly human-maintained (by me the user Zach). It will be collaborative where AI can make edits, but this doc esp the spec **must** be streamlined, simple and clear. I am not the most streamlined but often when coding agents maintain these design documents they get bloated and uninterpretable extremely quickly.
> This design doc should be the source of truth - if at any point you even a little not confident of how to execute the directions here, or that during coding you realise some conflict with existing design decisions, or you have the deviate from alignment with this design doc - PLEASE **CLARIFY** with me!! I'm okay with dumb questions, but I absolutely hate doing something that is not what the design doc set out to do. If you need to use the tool to ask me to choose between decisions, more than happy to do so.
> Since this design doc is so important - any changes even LLM made should be with approval from me (for e.g. in manual mode). Never write extensively or make extensive changes without my explicity approval of the content here (this advice mainly applies to the human edited part of this doc not agent context)
> In general this design doc (human maintained portion at least) should contain high level design decisions/ architecture of project etc. Small implementation details should be self-do
> If at any time you need to read the actual pdf/html arxiv papers to understand the methodology exactly, or need to read the actual codebase to understand whats gg on, or documentation online to check, pls do so!
> Gathering context is extremely important - at any time before coding - you should be sure you absolutely have ALL the relevant context needed to accomplish xxx task correctly. Don't worry - you have way more than enough context for this (I use 1M model always). If anything the strictest and only thing the user (I) do is purely context management - making sure you get the right and enough context. The worst thing you can do is, because you didn't gather the full context, you did something that did not integrate well with other code or didn't take into account core concepts. So make sure you always have enough, and the right, context before doing any tasks
> When editing these bits, ALWAYS have point form: make generous use of the bullets, sub-bullets, subsub-bullets, subsubsub-bullets etc. Each idea should be its own distinct point or line!!
> You will have quite alot of autonomy on this project to lead the directions and implementations etc. Should be clean clear code but we do value task progress
