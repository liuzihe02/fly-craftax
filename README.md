# fly-craftax
RL for FlyWire/MaleCNS fly to survive in open-ended Craftax environment

## Run

Everything runs from the repo root (the scripts resolve `data/` and `outputs/` relative to it).
On a pod without conda, pass the interpreter: `make test RUN=python3`.

- `make env setup data test` -- create the env, install the package, fetch the MaleCNS data, run the fast tests.
- `make bench` -- brain-only throughput benchmark.
- `make mn9` -- MN9 sugar-response check against Shiu et al.
- `make calibrate` -- sweep lamina bias and synapse weight, write `flycraftax/readout_norm.json`.
- `make eval` -- zero-shot rollout against the ablation controls, into `outputs/`.
- `make train` -- PPO on the linear DN readout, then a greedy evaluation against the M3 baselines (needs `make eval` first).
- `make viewer` -- render the scrubbable brain viewer: `outputs/viewer/index.html` and `outputs/viewer.gif`.
