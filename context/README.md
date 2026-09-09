# Agent context

Deep-research notes gathered by subagents on 2026-09-09. Reference material for design and implementation, not the spec. The spec lives in `../tracker.md`.

Each file ends with an "Open questions for spec" section. Those questions are the input to the spec design step.

- `malecns_data.md` - MaleCNS v1.0 file schemas, brain-vs-VNC filter, NT signing, edge-list thresholds, verified cell-type names and counts for every neuron in the I/O mapping
- `brain_model.md` - Shiu et al. 2024 LIF model: exact equations, parameters, Poisson drive, silencing, validated reference numbers, the fly-brain repo, JAX porting notes
- `craftax.md` - Craftax-Classic-Pixels internals: obs layout, state struct, action and achievement enums, movement and facing semantics, survival clocks, wrapper patterns, PPO baseline structure
- `doomfly.md` - prior MaleCNS + ViZDoom attempt: importer, retina mapping, DN readout, and the documented reasons it failed to learn
- `vision.md` - flygym compound-eye sampler, why it cannot be reused verbatim, two options for sampling a top-down frame, photoreceptor-to-column mapping in MaleCNS
- `bio_background_draft.md` - draft neuroscience background per circuit, pending owner review before any of it enters the spec
- `prior_art.md` - FlyGM, Digital Sphinx, embodied FlyWire, and other connectome-as-controller work

Local resources the notes refer to (both gitignored):

- `../external/` - clones of fly-brain, doomfly, Craftax, fly_connectome_data_tutorial, flygym
- `../data/` - MaleCNS body annotations and neurotransmitter feathers
