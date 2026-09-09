# Bio background (DRAFT, pending owner review)

## Photoreceptors and early vision

### R1-R6 (outer photoreceptors)
- What it is: the outer rhabdomere class, ~70-80% of rhabdomeres per ommatidium.
- Where it sits: terminate in the lamina, onto L1-L5, retinotopically organised (Rister et al., 2007, Neuron; Behnia and Desplan, 2015, Curr Opin Neurobiol).
- Function: broadband, high-sensitivity input driving motion and brightness pathways.
- Effect of activation: histaminergic, inhibitory onto lamina targets (all photoreceptors are histaminergic).
- Main inputs: direct phototransduction, no upstream synaptic input.
- Expected mapping direction: achromatic brightness/motion channel, drives pixel brightness input.

### R7/R8 (inner photoreceptors)
- What it is: one pair per ommatidium, inner rhabdomere class.
- Where it sits: pass through the lamina without synapsing, terminate in the medulla, R7 in layer M6, R8 in layer M3 (Morante and Desplan, 2008, Curr Biol; Schnaitmann et al., 2018, eLife, year unverified).
- Function: R7 is UV-sensitive (Rh3/Rh4), R8 is blue or green (Rh5/Rh6); pale (Rh3+Rh5) and yellow (Rh4+Rh6) ommatidia are stochastically distributed; R7 instructs R8 opsin choice (Mikeladze-Dvali et al., 2005; 2024 Development paper on functional opsin patterning).
- Effect of activation: histaminergic, inhibitory onto medulla targets.
- Main inputs: direct phototransduction.
- Expected mapping direction: colour/hue channel; colour discrimination biologically emerges from paired R7/R8 comparison, so picking R8 alone as the colour channel is a modelling simplification, not literal biology.

### Lamina neurons L1/L2
- What it is: the two principal lamina monopolar cells receiving R1-R6 input.
- Where it sits: lamina, one column per ommatidium.
- Function: L1 feeds the ON motion pathway, L2 feeds the OFF pathway; L1+L2 together are necessary and largely sufficient for motion behaviour (Rister et al., 2007, Neuron; Joesch et al., 2010, Nature).
- Effect of activation: downstream of R1-R6 inhibition, so their sign flips relative to photoreceptor drive.
- Main inputs: R1-R6.
- Expected mapping direction: not directly injected under the planned scheme (pixels go straight to R1-R6/R8), included here for completeness on the motion pathway.

### Medulla colour projection neurons
- What it is: neurons comparing R7 vs R8 signal within and across ommatidia (Karuppudurai et al., 2014, J Neurosci).
- Where it sits: medulla.
- Function: substrate for colour discrimination, built from the R7/R8 comparison rather than either channel alone.
- Effect of activation: not itemised in source material.
- Main inputs: R7, R8.
- Expected mapping direction: relevant if the spec later moves colour input downstream of R8 alone; flagged as an open question below.

## Hunger (NPF and alternatives)

### NPF neurons
- What it is: ~20 neurons in the protocerebrum, including a dorsomedial cluster.
- Where it sits: protocerebrum, signalling via NPFR1 onto dopaminergic mushroom-body-innervating neurons (Krashes et al., 2009, Cell).
- Function: activation mimics food deprivation in satiated flies, promotes appetitive memory expression and food-cue pursuit (Krashes et al., 2009, Cell); also suppresses sleep under starvation and modulates food choice rather than raw intake (Wu et al., 2003, Neuron, unverified exact citation; 2017 Cell Reports, "Drosophila NPF Signaling Independently Regulates Feeding and Sleep-Wake Behavior").
- Effect of activation: drives food-seeking behaviour; silencing blunts hunger-driven foraging.
- Main inputs: not itemised in source material beyond the general hunger/deprivation signal.
- Expected mapping direction: NPF drive scales up food-seeking; planned target for the hunger state.

### sNPF (alternative)
- What it is: a distinct peptide from NPF, despite the similar name.
- Where it sits: not itemised in source material.
- Function: controls sweet taste sensitivity (Frontiers 2019 review).
- Effect of activation: not itemised in source material.
- Main inputs: not itemised in source material.
- Expected mapping direction: alternative hunger-adjacent channel, not the primary pick.

### AKH-responsive neurons (alternative)
- What it is: 4 identified neurons responsive to adipokinetic hormone.
- Where it sits: not itemised in source material.
- Function: promote sugar consumption and restrict water consumption (Jourjine et al., 2016, Cell).
- Effect of activation: shifts consumption balance toward sugar, away from water.
- Main inputs: AKH signalling (metabolic state).
- Expected mapping direction: alternative for hunger, but its water-suppressing side effect overlaps the thirst channel, so it is a coupled rather than clean hunger signal.

### Sugar GRNs Gr5a/Gr64f (alternative)
- What it is: gustatory receptor neurons for sugar.
- Where it sits: labellum and legs (contact chemosensory).
- Function: fast contact detection, drives proboscis extension response (PER).
- Effect of activation: immediate feeding-initiation signal rather than a sustained hunger drive.
- Main inputs: direct sugar contact.
- Expected mapping direction: closer to a food-contact sensor than an internal hunger state; not the primary pick.

## Thirst (ppk28 GRNs, ISNs, thirst interneurons)

### ppk28 water GRNs
- What it is: water-sensing gustatory receptor neurons in labellar sensilla and tarsi.
- Where it sits: labellum, tarsi; express the PPK28 osmosensitive DEG/ENaC channel.
- Function: loss of PPK28 abolishes water response; activation triggers PER and drinking even without water present (Cameron et al., 2010, Nature; Chen et al., 2010).
- Effect of activation: drives water-seeking and drinking.
- Main inputs: direct osmotic sensing of water contact; also modulates lifespan via AKH (PNAS 2014, unverified relevance).
- Expected mapping direction: ppk28 GRN drive increases water approach; planned target for the thirst state as a sensory-level injection point.

### ISNs (internal-state / hemolymph osmolality sensors)
- What it is: a pair of SEZ neurons expressing Nanchung TRPV.
- Where it sits: subesophageal zone (SEZ).
- Function: sense hemolymph osmolality; high osmolality inhibits ISNs, which promotes drinking and suppresses feeding; AKH activates them (Jourjine et al., 2016, Cell).
- Effect of activation: inhibitory logic, so ISN activity is low when the fly is dehydrated.
- Main inputs: hemolymph osmolality, AKH.
- Expected mapping direction: a more literal interoceptive target than ppk28 if the goal is to inject internal thirst state directly, since ISNs sit downstream of the sensor rather than at it.

### Thirst interneurons (downstream)
- What it is: interneurons downstream of ISNs and ppk28.
- Where it sits: not itemised in source material beyond downstream of SEZ sensors.
- Function: promote water seeking and limit feeding (Landayan et al., 2021, eLife); D-serine gliotransmission promotes thirst behaviours (Chen et al., 2022, Curr Biol, unverified circuit position).
- Effect of activation: pro-thirst, anti-feeding.
- Main inputs: ISNs, glial D-serine signalling.
- Expected mapping direction: candidate integration point between the raw sensor (ppk28) and the interoceptive signal (ISNs).

## Fatigue and sleep (dFB, R5/ER5, the dFB controversy)

### dFB (dorsal fan-shaped body neurons)
- What it is: dorsal-layer fan-shaped body neurons, classic driver 23E10-GAL4 labels ~15-20 neurons.
- Where it sits: dorsal fan-shaped body, central complex.
- Function: activation induces sleep; long treated as the output arm of the sleep homeostat (Donlea et al., 2011, Science; Donlea et al., 2014, Science); cv-c Rho-GAP transduces sleep pressure into dFB excitability; circadian neurons excite dFB, dopaminergic arousal neurons inhibit it.
- Effect of activation: produces quiescence and raised arousal threshold, i.e. a gating state, not a specific motor pattern.
- Main inputs: circadian neurons (excitatory), dopaminergic arousal neurons (inhibitory), sleep-pressure signal via cv-c.
- Expected mapping direction: fatigue drive raises dFB excitability, gates locomotion/action outputs toward a sleep state.

### R5/ER5 ellipsoid body ring neurons
- What it is: a ring-neuron subtype of the ellipsoid body.
- Where it sits: ellipsoid body, central complex.
- Function: necessary and sufficient for sleep rebound; deprivation raises R5 calcium, BRP, and firing; act as a sleep-pressure integrator upstream of dFB (J Neurosci 2023, "Subtype-Specific Roles of Ellipsoid Body Ring Neurons"; "Circadian programming of the ellipsoid body sleep homeostat", 2022).
- Effect of activation: increases sleep pressure signal fed toward dFB.
- Main inputs: sleep deprivation history, circadian signals.
- Expected mapping direction: candidate site for injecting a fatigue/time-awake counter, upstream of the dFB gate.

### The dFB controversy
- De, Wu, Lambatan, Hua, Joiner (2023, Current Biology) found 23E10 dFB activation neither necessary nor sufficient for sleep, attributing prior phenotypes to VNC sleep-promoting (VNC-SP) neurons captured by the same driver line.
- Jones et al. (2025, PLOS Biology), using a refined dFB-Split driver, found dFB neurons do promote sleep but need stronger stimulation (>= 10 Hz), and are neurochemically mixed (glutamate + acetylcholine), not GABAergic as long assumed.
- Net read: the dFB-sleep link survives, but the 2011-2020 evidence base was confounded by off-target VNC neurons in the 23E10 driver.
- Practical consequence: a dFB-only fatigue-to-sleep mapping may be incomplete without a VNC-SP-equivalent pathway, which the MaleCNS/mapping has not identified yet.

## Descending neurons (counts, DNp09, DNa01/DNa02, MDN, halting, MN9 pathway, aggression circuits)

### DN counts
- Namiki et al. (2018, eLife) identified 190 bilateral pairs (>= 98 types) by light microscopy.
- The MANC connectome (Cheong et al., Marin et al., 2023-2025, eLife) found 1,328 DNs, far more than light-microscopy estimates.
- Implication: the connectome-level DN population is much larger and more heterogeneous than the classically named types the spec references.

### DNp09
- What it is: a descending neuron type.
- Where it sits: receives input from courtship neurons and visual projection neurons.
- Function: drives forward walking plus ipsilateral turning, necessary for male courtship pursuit (Bidaye et al., 2020, Neuron, not Cell).
- Effect of activation: forward locomotion with ipsilateral turn bias.
- Main inputs: courtship circuitry, visual projection neurons.
- Expected mapping direction: planned "forward" output channel.

### DNa01/DNa02
- What it is: a pair of descending neuron types associated with turning.
- Where it sits: largely non-overlapping input populations from one another.
- Function: activity predicts ipsilateral turning; left-right asymmetry correlates with rotational velocity. DNa02 produces transient turns via direct PFL3 input (heading vs goal comparison) and makes more direct motor-neuron connections, shortening the inside-turn stride; DNa01 produces slower, sustained turns (Rayshubskiy et al., 2024/2025, eLife; Cheong/Aimon et al., 2024, Cell).
- Effect of activation: ipsilateral turning, DNa02 faster/transient, DNa01 slower/sustained.
- Main inputs: DNa02 from PFL3 (heading-vs-goal); DNa01 inputs largely distinct from DNa02's.
- Expected mapping direction: planned "turn" output channel; owner should decide whether to fuse DNa01+DNa02 into one turn signal or keep them as fast/slow sub-channels.

### MDN (moonwalker descending neuron)
- What it is: the "moonwalker" descending neuron.
- Where it sits: not itemised in source material beyond its descending role.
- Function: activation sufficient for backward walking; silencing blocks backward walking at obstacles (Bidaye, Machacek, Wu, Dickson, 2014, Science).
- Effect of activation: backward walking.
- Main inputs: not itemised in source material.
- Expected mapping direction: planned "backward" output channel.

### Halting (no single braking DN)
- Sapkal et al. (2024, Nature 634:191-200) describe two separate halting mechanisms: "walk-OFF" GABAergic neurons that inhibit walking-command DNs during feeding, and "brake" cholinergic VNC neurons that arrest stepping during grooming.
- There is no single braking/halting DN; halting is a distributed, context-dependent mechanism.
- Implication: an "idle"/"stop" action mapped to a single DN would be a simplification; the real circuitry uses at least two distinct, behaviour-specific stopping pathways.

### MN9 pathway (SEZ to MN9)
- What it is: a motor neuron downstream of gustatory/feeding circuitry via the SEZ.
- Where it sits: subesophageal zone to MN9.
- Function: Shiu et al. (2024, Nature) calibrated a connectome weight scale so that 100 Hz sugar GRN drive gives ~80% of max MN9 firing; unilateral GRN activation drives contralateral MN9 more strongly than ipsilateral.
- Effect of activation: proboscis/feeding motor output.
- Main inputs: sugar GRNs via SEZ.
- Expected mapping direction: planned "do" action channel, alongside pC1/aIPg (see aggression note below); the Shiu calibration is a useful reference point for setting overall connectome weight scale, covered in context/brain_model.md.

### Aggression circuits (pC1/aIPg) — male/female correction
- The spec's shorthand "pC1/aIPg = male attack pathway" is not quite right per the source material.
- Schretter et al. (2020, eLife 9:e58942): aIPg and pC1d drive FEMALE aggression, not male.
- Chiu, Hoopfer, Coughlan, Pavlou, Goodwin, Anderson (2021, Cell): CAP neurons drive a shared aggressive-approach state in both sexes, MAP neurons drive male attack specifically, fpC1 drives female attack.
- Hoopfer et al. (2015, eLife): P1/pC1 male neurons drive a persistent aggressive state at sub-courtship activation thresholds.
- For a male fly model, the correct citations are Hoopfer et al. (2015, eLife) and Chiu et al. (2021, Cell); Schretter et al. (2020, eLife) should be cited only as the female homolog pathway, not the primary male circuit.
- MaleCNS-specific note: the dataset carries aIPg1-10 and aIPg_m1-4 (56 neurons total) and 49 pC1_* types (156 neurons), so the "aggression" label spans a large, heterogeneous population, not one clean cell type.
- Effect of activation: promotes aggressive approach/attack states, sex-dependent identity of the driving subtype.
- Main inputs: not itemised in source material beyond the P1/pC1 sub-courtship-threshold activation note.
- Expected mapping direction: candidate contributor to the "do" action alongside SEZ-to-MN9, but owner should decide whether to include aggression circuits at all for a male-only model, and if so, use MAP/CAP (Chiu 2021) and P1/pC1 (Hoopfer 2015) rather than the originally planned pC1/aIPg framing.

## Why a body-less brain with injected states is a strong simplification

- Neuromodulators act by volume transmission and reconfigure fixed circuits rather than acting as simple point-to-point wires (Marder and Bucher, 2001, Curr Biol; Bucher and Marder, 2013); injecting a scalar hunger/thirst/fatigue value at one cell type skips this reconfiguration dynamic entirely.
- Octopamine gates Gr32a-linked aggression and courtship circuits in males, another example of state acting on circuit configuration rather than as a direct drive signal.
- The ventral nerve cord (VNC) holds locomotor central pattern generators, e.g. a 3-neuron E1-E2-I1 CPG downstream of DNg100 (Pugliese et al., 2025/2026, bioRxiv); DN activity alone underdetermines limb coordination, since MANC and the female VNC connectome (Nature 2024) show 23,000+ VNC neurons that translate descending commands into coordinated motor output.
- A body-less Craftax agent has no VNC, no legs, no proprioception, and no muscle-level feedback loop, so descending-neuron output is read directly as a discrete action rather than as a command into a spinal-cord-like pattern generator.
- Injecting hunger/thirst/fatigue as constant or slowly-varying drive at a single cell type (NPF, ppk28, dFB) bypasses the real sensory transduction chain (gut stretch receptors, hemolymph osmolality sensors, sleep-pressure accumulation) that normally produces those signals.
- Net position: this project uses the connectome for its wiring/weights, not as a faithful whole-organism simulation; state injection and direct DN-to-action decoding are acknowledged shortcuts, not claims of biological completeness.

## Open questions for spec

- Colour channel: use R8 alone as planned (simplification), or model the R7/R8 paired comparison for a more biologically grounded colour signal?
- Hunger channel: inject at NPF (as planned), or consider AKH-responsive neurons or sNPF, given AKH's coupling to water consumption?
- Thirst channel: inject at ppk28 (sensory-level, as planned) or at ISNs (interoceptive-level, downstream of the real sensor)?
- Fatigue/sleep channel: keep the dFB-only mapping given the dFB controversy, or also model an R5/ER5-style sleep-pressure integrator upstream of dFB?
- Turn output: fuse DNa01+DNa02 into a single turn channel, or keep them as separate fast/slow turn sub-channels?
- Halting/idle action: accept a single simplified "stop" mapping despite there being no single biological braking DN, or model the two distinct walk-OFF/brake pathways?
- "Do" action: use SEZ-to-MN9 alone, or also include aggression circuitry (and if so, MAP/CAP per Chiu 2021 and P1/pC1 per Hoopfer 2015, not the originally planned pC1/aIPg framing)?
- Citation risk flags to resolve before finalising: Wu et al., 2003, Neuron citation was reconstructed from memory and needs verification; Schnaitmann et al., 2018 eLife year is unverified/conflicting; ppk28-AKH lifespan link (PNAS 2014) relevance to this project is unverified; D-serine gliotransmission circuit position (Chen et al., 2022, Curr Biol) is unverified.
- Scope: confirm whether L1/L2 and medulla colour-projection neurons stay purely as background context, or whether the mapping should route through them rather than injecting directly at R1-R6/R8.
