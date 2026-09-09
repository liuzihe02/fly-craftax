# MaleCNS v1.0 Connectome Data

Reference notes for building the connectome loader. All numbers below were computed
by running code against the local files, or by HTTP range-reading the remote edge list.

## Sources and files

- Bucket root: `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`
  - public, no login, CC-BY (stated on <https://male-cns.janelia.org/download/>)
- Files in that directory, exact sizes from the GCS JSON API:
  - `body-annotations-male-cns-v1.0-minconf-0.5.feather` 14.48 MB (local: `data/body-annotations.feather`)
  - `body-neurotransmitters-male-cns-v1.0.feather` 43.28 MB (local: `data/body-neurotransmitters.feather`)
  - `body-stats-male-cns-v1.0-minconf-0.5.feather` 778 MB, 88,384,522 rows, per-segment synapse counts
  - `connectome-weights-male-cns-v1.0-minconf-0.5.feather` 1051 MB, 151,856,684 rows
  - `connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather` 508 MB, 25,563,197 rows
  - `connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather` 502 MB, 25,568,639 rows
  - `syn-partners-*` (3.0-6.8 GB) and `syn-points-*` (13 GB) are per-synapse; not needed
  - `tbar-neurotransmitters-male-cns-v1.0.feather` 2.65 GB, per-tbar NT; not needed
- Listing command that produced the above:
  ```bash
  curl -s "https://storage.googleapis.com/storage/v1/b/flyem-male-cns/o?prefix=v1.0/connectome-data/&fields=items(name,size),nextPageToken"
  ```
- `minconf-0.5` refers to the synapse-detection confidence cut used to build the tables, not to an edge-weight threshold.
- Publication: Berg et al. (2025) bioRxiv, <https://www.biorxiv.org/content/10.1101/2025.10.09.680999v1>
- Do not confuse with the SJCABS harmonized `malecns_09` release (v0.9, BANC schema, different column names). See "Tutorial repo" below.

## Annotation file

- Local path: `/home/flowingpurplecrane/personal/fly-craftax/data/body-annotations.feather`
- Shape: 211,577 rows x 36 columns, one row per body (segment), not one row per neuron.
- `bodyId` is `int64`, no duplicates, range 10,001 to 1,571,825,087.
  - Note the range: these are dense small ints for well-traced neurons and large ints for fragments. Do not assume 64-bit FlyWire-style root ids.

### Column meanings

- `bodyId`: segment id. Joins to `body_pre` / `body_post` in the edge list and `body` in the NT file.
- `superclass`: the coarse anatomical/functional bucket. This is the column to filter on for "brain vs VNC".
- `class`: mid-level class, only populated for 26,513 of 211,577 rows (mostly sensory and mushroom-body / central-complex classes). Not a general-purpose partition.
- `subclass`: finer sensory detail (sensillum type, body part). 49 values, 89.6% null.
- `type`: the cell-type name, 11,751 distinct, 47,071 null. This is the label the I/O mapping should key on.
- `instance`: `type` plus a side suffix, e.g. `Tm12_R`, `KCg-m_R`, `DNpe008(PS225)_L`. 50,071 null.
- `flywireType`: matched FlyWire/FAFB cell type, 8,199 distinct, 68,421 null. Comma-separated when one MaleCNS type maps to several FlyWire types (e.g. `FB3B,FB3C,FB3E`). This is the bridge column for the FlyWire replication.
- `hemibrainType`: matched hemibrain type, 4,495 distinct, 178,658 null (hemibrain is half a brain, so coverage is low).
- `mancType`: matched MANC (male VNC) type, 3,893 distinct, 188,833 null. Mostly VNC and DN rows.
- `somaSide`: `L` / `R` / `M`, null for 60,851 rows. Null for essentially all photoreceptors (their somata are in the missing retina).
- `rootSide`: side of the neurite root where there is no soma in the volume. Populated for 17,939 rows, mostly sensory / photoreceptors / VNC afferents. Values `L`, `R`, `unknown`.
- `somaNeuromere`: neuromere containing the soma, only populated for 21,820 rows: `T1/T2/T3`, `A1..A10` (VNC), `CG`, `DC`, `TC`, `MD`, `MX`, `LB`, `GNG` (gnathal/SEZ). Null for the whole central brain and optic lobe, so it is NOT usable as a brain/VNC filter.
- `status`: coarse proofreading state. `Traced` 165,122, `Orphan` 15,925, `Glia` 11,864, `Unimportant` 10,751, null 5,472, `Assign` 1,832, `Anchor` 611.
- `statusLabel`: the fine 30-level ordered categorical behind `status` (`Reviewed`, `Roughly traced`, `Prelim Roughly traced`, `RT Hard to trace`, `Out of scope`, `Orphan-artifact`, ...). Stored as a pyarrow dictionary with `int8` indices.
- `assignedOlHex1` / `assignedOlHex2`: optic-lobe hexagonal column coordinates. Populated for 23,720 rows, all `ol_intrinsic`. Not populated for photoreceptors, so they cannot be used directly to build a retinotopic map from R cells.
- `group` / `supertype` / `serialMotif` / `mcnsSerial` / `mancSerial`: grouping ids for morphologically identical neurons and for serially homologous sets.
- `itoleeHl` / `trumanHl`: developmental hemilineage under two naming schemes.
- `birthtime`: `early` / `late`, 7,904 rows.
- `dimorphism`: `male-specific` 1,258, `sexually dimorphic` 771, plus `potentially_*`.
- `fruDsx`: fruitless / doublesex expression (`fru_high`, `dsx_high`, `coexpress_*`), 5,012 rows.
- `receptorType`: only three values, all pheromone GRNs (`putative_ppk23`, `putative_ppk25`, `putative_IR52b`). There is no water/sugar/bitter receptor annotation.
- `entryNerve` / `exitNerve`: peripheral nerve for sensory/motor neurons.
- `somaLocation` / `tosomaLocation`: `list<int64>` of length 3, xyz in native MaleCNS voxel space. Note the list dtype, not three scalar columns.
- `synonyms`: literature cross-references, e.g. `Cachero 2010: aIP-g; Yu 2010: pIP6`. Free text, worth grepping when a classic name is missing from `type`.
- `vfbId`, `mancBodyid`, `mancGroup`, `matchingNotes`: cross-dataset bookkeeping.

### superclass value counts (all 211,577 rows)

- `ol_intrinsic` 89,403
- null 44,877 (glia, orphans, unimportant fragments)
- `cb_intrinsic` 32,164
- `vnc_intrinsic` 13,161
- `visual_projection` 9,201
- `vnc_sensory` 6,370
- `ol_sensory` 6,098
- `cb_sensory` 4,868
- `ascending_neuron` 1,846
- `descending_neuron` 1,314
- `vnc_motor` 708
- `visual_centrifugal` 563
- `sensory_ascending` 537
- `cb_motor` 107
- `vnc_efferent` 94
- `cb_endocrine` 72
- `ENS` 50
- `vnc_tbc` 38, `vnc_sensory_tbc` 36, `vnc_endocrine` 22, `cb_sensory_tbc` 14, `sensory_descending` 12, `efferent_ascending` 8, `cb_efferent` 4, `efferent_descending` 4, `visual_projection_tbc` 2, `sensory_ascending_tbc` 2, `descending_neuron_tbc` 2

### class value counts (only 26,513 non-null)

- `visual` 6,091, `Kenyon_Cell` 4,064, `CX` 2,950, `olfactory` 2,639, `mechanosensory_tactile` 2,558, `mechanosensory` 1,733, `unknown_sensory` 1,712, `mechanosensory_proprioceptive` 1,454, `gustatory` 1,428, `ALPN` 686, `ALLN` 420, `DAN` 340, `ol_bilateral` 116, `MBON` 97, `hygrosensory` 66, `chemosensory` 58, `SEZPN` 27, `thermosensory` 25, `ALIN` 24, `ALON` 14, `mechanosensory_tbc` 11

### Selecting "optic lobe + central brain, drop VNC"

- The right column is `superclass`. Everything VNC-resident is prefixed `vnc_`, so a single prefix test is enough.
- Verified counts:
  - `superclass` not null and not starting with `vnc_`: **146,271**
  - the same, additionally `status == "Traced"`: **144,256**
  - `superclass in {cb_intrinsic, ol_intrinsic}` only: **121,567**
- **The spec's "~120k" does not match the brain-wide filter.** 120k is what you get if you keep only the two `*_intrinsic` classes and discard every sensory neuron, every visual projection neuron, and every DN. Since the project needs photoreceptors in and DNs out, the honest number is **~146k neurons, ~144k if traced-only**.
- `ascending_neuron` (1,846) has somata in T1/T2/T3/A* but arborises in the brain. It is kept by the `not vnc_*` rule. With the VNC gone these neurons have almost no input, so they will sit silent. Keeping or dropping them is a spec decision, not a data question.
- Verified snippet:
  ```python
  import pandas as pd

  ann = pd.read_feather("data/body-annotations.feather")
  is_vnc = ann.superclass.fillna("").str.startswith("vnc")
  brain = ann[ann.superclass.notna() & ~is_vnc]            # 146,271
  brain_traced = brain[brain.status == "Traced"]           # 144,256
  ```
- Side resolution, verified to cover all but 69 of the 146,271 brain rows:
  ```python
  side = brain.somaSide.fillna(brain.rootSide)
  # R 73,915 | L 71,776 | unknown 413 | M 98 | NaN 69
  ```
  Falling back further to the `_L` / `_R` / `_M` suffix of `instance` only recovers 2 more rows, so it is not worth the code.

## Neurotransmitter file

- Local path: `/home/flowingpurplecrane/personal/fly-craftax/data/body-neurotransmitters.feather`
- Shape: 1,835,518 rows x 10 columns. One row per body, all `body` values unique.
  - It covers every segment with synapses, so it is **8.7x larger than the annotation file**. Only 187,016 of the 211,577 annotated bodies appear in it (glia and synapse-free fragments do not).
- Columns:
  - `body` (`int64`): joins to `bodyId`.
  - `total_nt_predictions` (`int32`): number of tbars used for this body's prediction. Median 135 for brain neurons, max 31,347, min 0.
  - `predicted_nt` (str): per-body argmax NT.
  - `predicted_nt_confidence` (float): 0-1, 857 nulls file-wide.
  - `ground_truth` (str): experimentally known NT, populated for 85,484 bodies file-wide.
  - `cell_type` (str): the type name used for the type-level aggregate; null for 1,671,072 rows.
  - `celltype_total_nt_predictions`, `celltype_predicted_nt`, `celltype_predicted_nt_confidence`: the same prediction pooled over all members of the cell type. More robust for small/low-tbar neurons.
  - `consensus_nt` (str): the curated final call, combining per-body, per-type and ground truth. **Use this one.**
- Predicted NTs are the seven-way Eckstein/Bates set plus `unclear`: acetylcholine, gaba, glutamate, dopamine, serotonin, octopamine, histamine.
- Counts of `consensus_nt` over the 146,271 brain (non-VNC) neurons:
  - acetylcholine 92,384
  - glutamate 26,878
  - gaba 16,733
  - histamine 7,889
  - `unclear` 1,778
  - dopamine 392
  - null (no row in NT file) 134
  - octopamine 52
  - serotonin 31
- Signing:
  - Unambiguous: acetylcholine `+1`; gaba, glutamate, histamine `-1`.
  - **Ambiguous for signing: `unclear`, dopamine, serotonin, octopamine, and missing rows.** On the 144,256 brain traced neurons the breakdown is `unclear` 1,768, dopamine 392, missing 129, octopamine 52, serotonin 31, total **2,372, or 1.6%**.
  - Histamine matters here: all 6,091 photoreceptors are histaminergic and inhibitory onto L1-L3 / Tm neurons, so signing histamine as `-1` is correct and not optional.
  - `unclear` clusters in `cb_intrinsic` (798), `visual_projection` (481), `cb_sensory` (100), `cb_motor` (80), `cb_endocrine` (71).
- Verified snippet:
  ```python
  nt = pd.read_feather(
      "data/body-neurotransmitters.feather",
      columns=["body", "consensus_nt", "predicted_nt_confidence", "total_nt_predictions"],
  )
  SIGN = {"acetylcholine": 1.0, "glutamate": -1.0, "gaba": -1.0, "histamine": -1.0}
  df = brain_traced.merge(nt, left_on="bodyId", right_on="body", how="left")
  df["sign"] = df.consensus_nt.map(SIGN)
  # signed 141,884 | unsigned 2,372
  ```
- Spot checks: `R1-R6` histamine (conf 0.97), `R7*/R8*` histamine, `DNp09` / `DNa01` / `DNa02` / `MDN` acetylcholine (conf ~0.95), `MN9` acetylcholine (conf 0.53), `ER5` gaba, `FB6A_a` glutamate, `FB7B` unclear, `NPFL1-I` unclear, `aIPg1` and `pC1_1a` acetylcholine.

## Edge list

- Schema of `connectome-weights-male-cns-v1.0-minconf-0.5.feather`, read from the first 8 KB over HTTP:
  - `body_pre` `int64`, `body_post` `int64`, `weight` `int64`. Three columns, nothing else.
- `weight` is a raw **synapse count** (post-synaptic sites). Max observed 2,591.
- **Not thresholded.** 151,856,684 rows, minimum weight 1.
- **Rows are sorted by `weight` descending.** This is useful: a threshold cut is a prefix of the file, so you can stream just the head.
- `body_pre` / `body_post` are the same ids as `bodyId` in the annotation file. Confirmed by joining sampled batches.
- The file is Arrow IPC with 2,318 record batches of 65,536 rows.
- Two smaller variants add `type_pre` and `type_post` string columns:
  - `-traced-only` 25,563,197 rows, 508 MB. Only edges where both bodies are traced neurons.
  - `-significant-only` 25,568,639 rows, 502 MB. Practically the same set (5,442 more rows).
  - **Recommend downloading `-traced-only` (508 MB) instead of the full 1.1 GB file.** It already excludes orphans/fragments, carries the type labels, and loses only edges you were going to drop anyway.

### Edge counts at various thresholds (measured, not estimated)

| threshold | full file (151.86 M) | traced-only (25.56 M) |
|---|---|---|
| >= 1 | 151,856,684 | 25,563,197 |
| >= 2 | 57,670,765 | 15,270,273 |
| >= 3 | 23,014,406 | 10,511,038 |
| >= 5 | **7,622,864** | **6,235,682** |
| >= 10 | 2,799,910 | 2,749,407 |

- After additionally restricting both endpoints to the 146,271 brain (non-VNC) bodies, sampling 16 of the 96 batches that make up the `weight >= 5` prefix of the traced-only file gives 78.5% of `weight >= 5` rows surviving, i.e. **roughly 5.2 M brain-brain edges at weight >= 5**.
- Sizing consequence: 5.2 M edges as `int32` indices plus `float32` values is about 60 MB. A BCOO sparse matvec over 144k neurons is trivially within an 8 GB GPU. The threshold is a modelling choice, not a memory constraint.
- Reading remote batches without downloading the file (verified, uses only stdlib plus pyarrow):
  ```python
  import io, urllib.request
  import pyarrow.ipc as ipc

  class HttpFile(io.RawIOBase):
      def __init__(self, url):
          self.url, self.pos = url, 0
          req = urllib.request.Request(url, method="HEAD")
          self.size = int(urllib.request.urlopen(req).headers["Content-Length"])
      def readable(self): return True
      def seekable(self): return True
      def tell(self): return self.pos
      def seek(self, off, whence=0):
          self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
          return self.pos
      def read(self, n=-1):
          if n < 0: n = self.size - self.pos
          if n == 0: return b""
          end = min(self.pos + n, self.size) - 1
          req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
          data = urllib.request.urlopen(req).read()
          self.pos += len(data)
          return data

  BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
  r = ipc.open_file(HttpFile(BASE + "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"))
  print(r.schema, r.num_record_batches)
  batch0 = r.get_batch(0).to_pandas()
  ```

## Cell types for the I/O mapping

All counts are rows in the annotation file. Side is `somaSide` unless noted.

### Photoreceptors, present, naming differs from the spec

- `superclass == "ol_sensory"`, 6,098 rows, of which 6,091 are R-cells and 7 are `HBeyelet`.
- Exact `type` strings and counts:
  - `R1-R6` 3,377 (outer photoreceptors are pooled into ONE type, they are not split R1..R6)
  - `R7y` 482, `R7p` 332, `R7d` 82, `R7_unclear` 404
  - `R8y` 481, `R8p` 330, `R8d` 76, `R8_unclear` 442
  - `R7R8_unclear` 85
- Side: `somaSide` is null for 6,062 of 6,091 (no retina in the volume). Use `rootSide`: R 3,746, L 2,345, null 7. Or the `instance` suffix (`R1-R6_L`, `R7y_R`, ...), which is complete.
- `assignedOlHex1` / `assignedOlHex2` are **null for every photoreceptor**. They are populated only for 23,720 `ol_intrinsic` neurons. So a retinotopic ring/hex map cannot be built from the R cells directly; you would have to propagate hex coordinates from their postsynaptic lamina/medulla partners, or assign hex positions via `optic-lobe-column-pins` (`gs://flyem-male-cns/v1.0/malecns-v1.0-optic-lobe-column-pins/`).
- 1,983 of 6,098 `ol_sensory` rows have `status` null with `statusLabel == "Out of scope"`. Only 4,114 are `Traced`. Filtering on `status == "Traced"` throws away a third of the retina.
- `y` / `p` / `d` suffixes are yellow / pale / dorsal-rim ommatidia subtypes: `R7y`+`R8y` are the yellow pair, `R7p`+`R8p` pale, `R7d`+`R8d` dorsal rim (polarisation). For a brightness/colour split, `R1-R6` is brightness and `R7*/R8*` is colour, exactly as the spec assumes.

### Descending neurons

- `superclass == "descending_neuron"`: **1,314 neurons**, 480 distinct `type` values, 4 with null type. Side: L 656, R 648, M 10.
- This confirms the spec's "~1300 DNs". `superclass` is the column that marks them.
- 1,276 of 1,314 get an unambiguous NT sign.
- Named DNs verified present, all bilateral pairs:
  - `DNp09` 2 (1 L, 1 R)
  - `DNa01` 2 (1 L, 1 R)
  - `DNa02` 2 (1 L, 1 R)
  - `MDN` 4 (2 L, 2 R) — MaleCNS keeps all four moonwalkers under one `type`, unlike FlyWire which gives four root ids
- `MN9` exists as `type == "MN9"`, 2 neurons, `superclass == "cb_motor"` (not a DN). This is the proboscis motor neuron the spec wants for `do`.

### Courtship / aggression types

- `pC1`: 156 neurons across 49 `type` values, `pC1_1a` ... `pC1_19`, plus `pC1x_a..d`. All `cb_intrinsic`. There is no single `pC1` type string, so match with `type.str.startswith("pC1")`.
  - `synonyms` confirms the identity: `Lee 2002, Rideout 2010, Nojima 2021: pC1; Cachero 2010: pMP-e; Yu 2010: pMP4` on 86 rows.
- `aIP-g` **does not exist under that spelling**. The correct strings are `aIPg1`..`aIPg10` and `aIPg_m1`..`aIPg_m4`, 56 neurons total, 28 L / 28 R, all `cb_intrinsic`.
  - Beware: `synonyms` contains the classic Cachero 2010 `aIP-a` .. `aIP-h` names on 533 rows, which are a *different* and much larger set. Do not match on `synonyms` for aIPg.

### NPF, thin

- `type == "NPFL1-I"`, only **2 neurons** (1 L, 1 R), `cb_intrinsic`, `consensus_nt == "unclear"`.
- `hemibrainType == "NPFP1"` on 2 more, whose `instance` is `DNp29(NPFP1)_L/R` — these are descending neurons.
- That is the entire NPF footprint. There is **no neuropeptide column** in the v1.0 annotation file, so no way to find all NPF-expressing neurons from this table alone.
- Driving "hunger" through 2 neurons is very different from driving it through a population. Options: use `NPFL1-I` + `DNp29` (4 cells), or pick a larger hunger-related population by hand.

### dFB / sleep, not labelled as such

- The string `dFB` does not appear anywhere in the file. Neither does `23E10`, `helicon`, or `sleep`.
- Fan-shaped body tangential neurons are named `FB<layer><letter>`: 601 neurons over 166 types matching `^FB[0-9]`.
- The dorsal FB layers are 6 and 7: `^FB[67]` gives 140 neurons over 43 types, `FB6A_a/b/c`, `FB6B` ... `FB6Z`, `FB7A` ... `FB7M`. The canonical sleep-promoting dFB neurons in hemibrain terms are within `FB6` and `FB7` (`FB6A`, `FB6C`, `FB6H`, `FB7B` are the usual suspects), but MaleCNS does not tag which.
- `ExR1`..`ExR8` all exist, 26 neurons total. `ExR1` (4 neurons) is the dopaminergic ExR that innervates the EB and is sleep-relevant; note its `consensus_nt` here is acetylcholine, conf 0.76, which is worth a second look.
- `ER5` exists, 21 neurons (11 L, 10 R), gabaergic. ER5 is the best-characterised sleep-homeostat population in the ellipsoid body and is a cleaner, better-defined target than "dFB".
- `hDeltaK` exists (31 neurons) as do `hDeltaA`..`hDeltaM`, but hDelta neurons are columnar FB neurons, not the sleep-promoting tangentials. `hDeltaK` is not a sleep type.

### Water-sensing gustatory neurons, absent

- Nothing in the file matches `ppk28`, `water`, `sugar`, `bitter`, `Ir56d`, or `Gr5a` in any string column.
- `class == "gustatory"` is 1,428 neurons but 1,073 of them are `vnc_sensory` (leg and wing bristles) and get dropped with the VNC. Only **275 are `cb_sensory`**, plus 80 `sensory_ascending`.
- The head GRNs available are, by `subclass`:
  - `labellar bristle` 163, types `LB1a`..`LB4b`
  - `taste peg` 60, types `claw_tpGRN` (50), `dorsal_tpGRN` (10)
  - `pharyngeal sensillum` 48, types `PhG1a`..`PhG16`
- `receptorType` only distinguishes pheromone channels (`putative_ppk23` 269, `putative_ppk25` 257, `putative_IR52b` 225), all on leg/wing bristles.
- **There is no way to identify water GRNs from this file.** Routes forward:
  - the SJCABS harmonized `malecns_09_meta.feather` has `cell_function_detailed`, whose vocabulary (`data/meta_data_entries.csv` in the tutorial repo) includes `water`, `dry`, `humid`, `Gr5a_or_Ir60d`, `bitter_Gr33a` — but that is the v0.9 release with different ids
  - or pick a labellar `LB*` type by hand from the literature and treat it as the thirst channel
  - or use `class == "hygrosensory"` (66 neurons, `HRN_VP4` 28, `HRN_VP1d` 18, `HRN_VP5` 12, `HRN_VP1l` 8) as a humidity proxy for thirst

### Other types that may be useful

- `class == "hygrosensory"` 66, `thermosensory` 25 (`TRN_VP1m`, `TRN_VP2`, `TRN_VP3a/b`), `chemosensory` 58 (`SNch01`, `SNch10`)
- Octopaminergic `OA-*` types: 37 neurons over 16 types (`OA-VUMa1..a8`, `OA-VPM3/4`, `OA-ASM1..3`, `OA-AL2i1..i4`) — arousal / reward modulators
- Dopaminergic `PPL1*` 24, `PPM12*` 14

## Tutorial repo

- Local clone: `/home/flowingpurplecrane/personal/fly-craftax/external/fly_connectome_data_tutorial`
- Upstream: <https://github.com/sjcabs/fly_connectome_data_tutorial>
- **It does not read the v1.0 flat-connectome files we have.** It reads the SJCABS-harmonized re-release from a different bucket:
  - `gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/{dataset}_{version}/`
  - datasets: `banc_888`, `fafb_783`, `manc_121`, `hemibrain_121`, `malecns_09`
  - MaleCNS there is **v0.9**, 165,114 neurons, ids in `malecns_09_id`, harmonized column names (`super_class`, `cell_class`, `cell_sub_class`, `cell_type`, `region`, `side`, `neurotransmitter_predicted`, `cell_function_detailed`) — none of which match our v1.0 column names.
  - Its `region` column has exactly the four values `central_brain`, `optic_lobe`, `ventral_nerve_cord`, `neck_connective`, which is a cleaner brain/VNC split than our `superclass` prefix test, but only in v0.9.
- Reusable pieces in `python/utils.py` (957 lines):
  - `construct_path(data_root, dataset, file_type)` builds bucket paths
  - `read_feather_gcs` / `read_parquet_gcs` with a local `.cache/` directory — worth copying the caching pattern
  - `REGION_SPECS` and `subset_by_region(meta, dataset, region, ...)` reproduce region cut-outs. The `optic` spec is `neuropil` matching `^LO|^LOP|^AME|^ME` with >= 100 synapses; `suboesophageal_zone` is `^FLA|^SEZ|^GNG|^SAD|^AMMC|^PRW`. These operate on the *synapse* table, which we do not have, so they are reference logic rather than drop-in code.
  - `split_neurons_by_compartment`, `plot3d_split` for navis-based visualisation
- Class hierarchy vocabulary: `data/meta_data_entries.csv`, 248 rows, one column per metadata field listing every legal value. Useful as a controlled vocabulary but it is the BANC schema, not MaleCNS v1.0.
- Dataset docs: `data/dataset_documentation/{malecns,fafb,banc,manc,hemibrain}_data.md`.
- `python/fly_connectome_06_LIF_model.ipynb` is the **Shiu et al. LIF model on FlyWire**, in Brian2, and it already hardcodes the exact FlyWire root ids for the output neurons this project wants:
  - `P9_oDN1_left/right`, `P9_left/right` (forward velocity)
  - `DNa01_left/right`, `DNa02_left/right` (turning)
  - `MDN_1..MDN_4` (backward walking)
  - `Giant_Fiber_1/2` (escape)
  - `MN9_left/right` (proboscis motor, feeding)
  - `aDN1_left/right` (antennal grooming)
  - This is the single most directly reusable artefact in the repo, and it is the ready-made output mapping for the FlyWire replication.

### FlyWire v783 differences, brief

- FAFB/FlyWire v783 is brain only, female, ~140,177 neurons, ~69 M synapses, 15,023,799 edges. No VNC, so the "drop VNC" step disappears.
- Ids are `fafb_783_id`, 18-digit `int64` root ids (e.g. `720575940627787609`), a completely different id space from MaleCNS `bodyId`. The bridge is the `flywireType` column in the MaleCNS annotations, or the `hemibrain_121_cell_type` columns in the harmonized meta.
- FlyWire's edge list is `pre`/`post`/`count`/`norm`/`total_input`, so `count` plays the role of `weight`.
- Its own NT prediction lives in the meta table (`neurotransmitter_predicted`, `neurotransmitter_score`), not a separate file, and it is a six-way prediction without histamine confidence at the same quality — photoreceptors are absent from FlyWire anyway (no retina, and the lamina is only partly reconstructed).
- Practical consequence for the loader: abstract over (id column, edge column names, NT column location) and the same pipeline runs on both.

## Gotchas

- `bodyId` is `int64` and mixes small ids (10,001) with 10-digit ids. Never cast to `int32`, and never round-trip through `float` — `group`, `mancBodyid`, `mancSerial`, `mcnsSerial` are already stored as `double`, so those columns have lost exactness for large values.
- No duplicate `bodyId` in the annotation file, so `set_index("bodyId")` is safe.
- 44,877 annotation rows have a null `superclass`. These are glia (11,864), orphans (15,925), unimportant fragments (10,751) and assign/anchor stubs. Filtering `superclass.notna()` removes all of them in one step.
- The NT file has 1,835,518 rows but only 187,016 of them correspond to annotated bodies. Merge with `how="left"` from the annotation side, never the other way, or you will pull in 1.6 M fragments.
- 129-134 brain neurons have no row in the NT file at all (mostly `cb_sensory` 88 and `ENS` 37). Handle `NaN` sign explicitly.
- `total_nt_predictions` can be 0. Confidence for those is meaningless; consider a minimum tbar count before trusting `predicted_nt`, or just use `consensus_nt` which already falls back to the cell-type-level call.
- Traced vs fragment: `status == "Traced"` is 165,122 bodies file-wide. Within the brain filter it removes only 2,015 of 146,271, and 1,983 of those are photoreceptors marked `Out of scope`. **Applying `status == "Traced"` silently deletes a third of the retina.** Either skip the status filter for `ol_sensory`, or use the `-traced-only` edge list, whose definition of traced may differ from the annotation `status` column.
- `statusLabel` is an ordered pyarrow dictionary with 11 categories that have zero rows. `value_counts()` shows them as 0.
- Photoreceptor `somaSide` is null. Any code that groups by `somaSide` will drop the entire retina without warning.
- The edge list is sorted by descending weight, so a naive `head(n)` is a top-n-strongest-connections sample, not a random sample.
- `flywireType` and `hemibrainType` can be comma-separated multi-matches (`FB3B,FB3C,FB3E`, `DNg08_a,DNg08_b`). Split on `,` before joining.
- License: CC-BY. Cite Berg et al. (2025), <https://www.biorxiv.org/content/10.1101/2025.10.09.680999v1>. The NT predictions come from the Eckstein & Bates et al. (2024) classifier and are usually cited separately.

## Open questions for spec

- **Neuron count.** The spec says ~120k. The brain filter gives 146,271 (144,256 traced). Update the spec to ~146k, or explicitly narrow to `cb_intrinsic + ol_intrinsic` (121,567) and accept that photoreceptors and DNs are then outside the simulated population.
- **Ascending neurons.** Keep the 1,846 `ascending_neuron` cells as silent brain-resident nodes, or drop them since their VNC drive is gone?
- **Status filter.** Apply `status == "Traced"`? If yes, an exception is needed for `ol_sensory` or a third of the retina disappears.
- **Which edge file.** Download `-traced-only` (508 MB, has `type_pre`/`type_post`) or the full 1.1 GB file? The traced-only file drops 1.4 M of the 7.6 M `weight >= 5` edges, all involving untraced fragments.
- **Threshold.** `>= 5` gives ~5.2 M brain-brain edges. `>= 3` and `>= 2` are both affordable on the target GPU. Is 5 a considered choice or inherited from Shiu et al.?
- **Signing the 2,372 unsigned neurons.** Dopamine/serotonin/octopamine (475) are modulatory; `unclear` (1,778) and missing (134) are unknown. Options: drop their outgoing edges, treat as excitatory, treat as zero-weight, or fall back to `celltype_predicted_nt`. This needs a decision because it affects 1.6% of the network.
- **NPF hunger input.** Only 2 `NPFL1-I` neurons exist (plus 2 `DNp29`). Accept a 2-neuron input channel, or choose a different / broader hunger population?
- **dFB fatigue input.** MaleCNS has no `dFB` label. Pick from `FB6*` / `FB7*` tangentials (140 neurons, 43 types) by literature, or switch the fatigue channel to `ER5` (21 gabaergic neurons, well-defined sleep homeostat)?
- **Thirst input.** No water GRNs exist in this file. Fall back to `class == "hygrosensory"` (66 neurons), pick a labellar `LB*` type by hand, or cross-reference the v0.9 harmonized `cell_function_detailed == "water"` and map ids?
- **Retinotopy.** `assignedOlHex1/2` is null for all photoreceptors, so the ~800-per-eye ring retina cannot be built from R-cell metadata alone. Propagate hex coordinates from postsynaptic `ol_intrinsic` partners, use the optic-lobe column pins, or map pixels to R cells arbitrarily and let training sort it out?
- **`R1-R6` is one type, not six.** 3,377 outer photoreceptors with no per-subtype label and no column assignment. Decide how pixels map onto them.
- **aIPg naming.** The spec says `aIP-g`; the data says `aIPg1..10` and `aIPg_m1..m4`. Confirm the intended 56-neuron set.
- **Which `pC1` subtypes.** 49 distinct `pC1_*` types, 156 neurons. All of them, or a subset?
