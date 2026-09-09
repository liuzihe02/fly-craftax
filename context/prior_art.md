# Prior art: connectome-as-controller

## FlyGM
- arXiv 2602.17997, "Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly", Tsinghua, 2026.
- Fixed adult whole-brain connectome used as a graph-structured GNN policy.
- Signed synaptic weights are fixed; neurons split into afferent/intrinsic/efferent classes.
- Message passing is multiplication by the fixed signed weight matrix.
- Per-node MLPs are conditioned on trainable neuron descriptors, so the connectome topology stays fixed while node behaviour is learned.
- Sensory input is two 32x32 RGB MuJoCo eyes, 150 degree FOV, 4.6 degree inter-ommatidial angle.
- Efferent node states decode to joint commands.
- Two-stage training: imitation learning, then PPO fine-tuning.
- Result: stable gait initiation, walking, turning, flight, and better sample efficiency than baselines.
- Sources: arxiv.org/abs/2602.17997, alphaxiv overview.

What we can borrow:
- Fixed connectome backbone plus a small trainable readout/descriptor layer, rather than training the whole graph.

What they warned about or got wrong:
- This is a positive-results paper; it is unclear how much of the gain comes from the connectome specifically versus the generic graph inductive bias of a GNN policy.

## Digital Sphinx
- eLife reviewed preprint 111516, 2026, Brunton lab, University of Washington, "The digital sphinx: Can a worm brain control a fly body?".
- Fixed C. elegans 302-neuron connectome drives a simulated Drosophila body.
- Fly sensory feedback is routed into the worm connectome.
- Only the worm-motor-output to fly-leg-actuator interface is trained, via deep RL.
- Result: realistic walking despite the total anatomical mismatch between worm brain and fly body.
- Central argument: a flexible, learned interface does the explanatory work, so behavioural plausibility from a connectome-driven system can be a spurious signal of brain authenticity.
- Also covered in The Transmitter and PMC13041780.

What we can borrow:
- Constrain the trainable readout to as few free parameters as possible, so any resulting behaviour is attributable to the connectome rather than to interface capacity.

What they warned about or got wrong:
- Do not over-interpret behavioural plausibility alone as validation that the connectome is doing meaningful work.

## Embodied FlyWire in NeuroMechFly (erojasoficial-byte/fly-brain)
- github.com/erojasoficial-byte/fly-brain, Rojas Aliaga 2026, Zenodo preprint DOI 10.5281/zenodo.19152238, solo author.
- Full FlyWire v783 connectome (138,639 neurons, ~15.1M synapses), simulated as a GPU LIF model in PyTorch at 5 kHz.
- Embodied in NeuroMechFly v2 (87 joints, 721-ommatidia eyes).
- Visual, olfactory, gustatory, and somatosensory input; ~1,100 DNs feed a hand-built brain-body bridge (DN rates map to drive rates plus discrete mode selection: walk/escape/groom/feed/flight) that produces torques.
- Not RL-trained; uses Hebbian plasticity, dW = eta * r_i * r_j - alpha * W.
- Headline result: two identical connectomes diverge after 24h of embodied experience (81% vs 47% escape rate, 76,034 divergent synapses).
- Includes procedural_arena.py, a "Minecraft-like" arena, but not Craftax itself.
- Mature repo, not peer reviewed.

What we can borrow:
- The DN-rate-to-discrete-mode bridge is a useful decoding reference for turning DN firing rates into a small action set, similar to what this project needs for Craftax.

What they warned about or got wrong:
- Emergent-individuality claims drawn from Hebbian drift risk overclaiming; divergence between two runs is not on its own evidence of anything beyond stochastic dynamics.

## Other 2024-2026 work
- eonsystemspbc/fly-brain: a multi-backend speed benchmark of the Shiu model, no body attached.
- FLYNN (arXiv 2607.00025): a fly-topology-inspired robot navigation network, not a connectome simulation.
- Nature 2025 whole-body fly locomotion physics simulator (s41586-025-09029-4): uses a conventional neural-network controller, not a connectome.
- No Craftax, Crafter, or Minigrid connectome-driven-agent work was found as of September 2026.

What we can borrow:
- Nothing directly usable from these three (no connectome, or no body/environment), but they confirm the design space nearby is sparse.

What they warned about or got wrong:
- Not applicable; these are adjacent, not comparable, efforts.

## Shiu et al. 2024
- Nature 634, 210-219; repo github.com/philshiu/Drosophila_brain_model.
- Covered in depth separately in context/brain_model.md; not re-summarised here.

## Open questions for spec

- Backbone approach: fixed connectome plus trainable per-node descriptors (FlyGM-style), or fixed connectome plus Hebbian plasticity (Embodied FlyWire-style), or something else?
- Interface size: how many free parameters should the sensory-injection and action-decoding layers carry, given Digital Sphinx's warning that a flexible interface can explain away the connectome's contribution?
- Action decoding: discrete mode selection from DN rates (Embodied FlyWire-style bridge), or direct continuous decoding into Craftax's action space?
- Novelty framing: since no Craftax/Crafter/Minigrid connectome work was found as of Sept 2026, confirm whether this search was thorough enough to state novelty in any write-up, or whether it should be framed as "not found in our search" rather than "does not exist".
