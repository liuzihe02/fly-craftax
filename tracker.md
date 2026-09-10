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
- JAX, sparse BCOO matvec, batched over envs, fits the 8 GB 4060; steps per action fixed after a throughput benchmark
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

Outputs (fixed mapping, zero-shot)
- forward: DNp09; backward: MDN; turn: DNa02 + DNa01 left-minus-right; do: MN9; sleep: FB6/FB7 tangentials
- action = argmax of signals, noop when all below a floor
- Ablation controls built in from day one: black frame, shuffled weights, disconnected inputs

Trained alternative: brain frozen, linear readout over 1,314 DN rates to 7 logits, PPO

### Viewer

All are panels

- Craftax visualization
  - Want to see fly as sprite and any other realistic simulations
- Real-time interactive visualization using the existing libraries to see the activations
- Other indicators of the fly brain state or anything? TBD

### Milestones

- [ ] M1 loader + JAX kernel + Brian2 validation, plan: `docs/plans/2026-09-09-m1-brain.md`
  - [x] scaffold, pinned env, data download
  - [x] connectome loader with cache
  - [x] Brian2 oracle
  - [x] JAX LIF kernel, spike-for-spike match on toy graphs
  - [x] oracle match on a 2000-neuron MaleCNS subgraph
  - [x] full-brain throughput benchmark on the 4090
    - 4090: 3,817 steps/s at batch 1 (0.38x realtime per env); batch 32 gives 152 steps/s and 4,851 env-steps/s, 0.98 GiB peak. Delay buffer as a ring beat the concatenate by 7% (batch 1) to 22% (batch 32) and is kept; BCSR, segment_sum, a transposed buffer and a driven-subset RNG were all equal or slower.
  - [ ] MN9 smoke check with labellar GRN drive
- [ ] M2 Craftax wrappers + retina + state encoders, plan: `docs/plans/2026-09-10-m2-env.md`
  - [ ] craftax installed and pinned, baseline wrappers copied, one env step
  - [ ] egocentric action wrapper
  - [ ] survival reward wrapper and batched env stack
  - [ ] photoreceptor retinotopy from lamina and medulla synapses
  - [ ] radial retina sampler rotated by facing, visual check
  - [ ] drive assembly: retina plus hunger, thirst, fatigue
  - [ ] integration rollout through env, drive, brain, visual check
- [ ] M3 zero-shot loop with ablation controls
- [ ] M4 PPO linear readout
- [ ] M5 viewer

## Known Limitations

- No neuromodulation: hunger, thirst, fatigue injected as spikes, not emergent
- No body, no VNC; DN activity read as discrete actions
- Zero baseline activity: the brain is silent unless driven, so inhibition is invisible on silent neurons
- Locomotion DN mapping is from optogenetics and hobby projects, not validated in the Shiu model
- Hunger is a 2-cell channel; thirst and fatigue use proxy cell types since water GRNs and dFB are unlabelled in MaleCNS
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
