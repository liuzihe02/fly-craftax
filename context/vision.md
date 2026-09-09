# flygym compound-eye vision module

Source: `external/flygym` (NeLy-EPFL/flygym, cloned local copy, matches upstream `main` structure verified via GitHub API). Version `2.1.0` per `pyproject.toml`. License **Apache-2.0** (repo `LICENSE`, `pyproject.toml: license = "Apache-2.0"`); the fisheye-correction algorithm inside it is adapted from a separate MIT-licensed project (`Gil-Mor/iFish`), noted in the docstring at `retina.py:200-202`.

## Module layout

- Only one real source file: `src/flygym/vision/retina.py` (276 lines).
  - `src/flygym/vision/__init__.py` is empty — nothing re-exported, you import `from flygym.vision.retina import Retina` directly (that's what `simulation.py` does).
  - Single class `Retina`, no submodules, no other classes.
- Data assets actually loaded by `Retina.__init__` (`retina.py:78-99`):
  - `src/flygym/assets/model/neuromechfly/vision.yaml` (874 bytes) — scalar config: `fovy_per_eye`, `raw_img_height_px`, `raw_img_width_px`, `num_ommatidia_per_eye`, `fisheye_distortion_coefficient`, `fisheye_zoom`, plus `hidden_segments` (body parts hidden from the eye cameras) and `sensors` (camera parent body, relative position/orientation, marker colour).
  - `src/flygym/assets/model/neuromechfly/compound_eye.npz` (27,591 bytes, compressed) — the actual retina-mapping data:
    - `ommatidia_id_map`: int64 array, shape `(512, 450)` — matches `raw_img_height_px x raw_img_width_px`. Each entry is the 1-indexed ommatidium ID that raw pixel belongs to; `0` = outside the hex lattice (background).
    - `pale_mask`: bool array, shape `(721,)` — pale-type (`True`, 216 of 721 ≈ 30%) vs yellow-type (`False`, 505 of 721 ≈ 70%) per ommatidium.
  - `src/flygym/assets/model/flybody/vision.yaml` — same four scalar values (`fovy_per_eye=157`, 512x450, 721 ommatidia) but different camera `parent`/`rel_pos`/`orientation`, for the alternate "flybody" mesh variant. It reuses the *same* `compound_eye.npz`, i.e. eye optics are treated as identical across body models, only head mounting geometry differs.
  - **Orphaned duplicate data**: `src/flygym/assets/model/neuromechfly/vision/ommatidia_id_map.npy` (460,928 bytes, uint16, same `(512,450)` shape) and `.../vision/pale_mask.npy` (849 bytes) sit next to `compound_eye.npz` but are **not referenced by any code** in the repo (`grep` for their path/filename found nothing) — `Retina.__init__` only ever loads `compound_eye.npz`. Worth treating this `vision/` subfolder as stale/leftover, not a second data source.
- Generator (not shipped, but explains where the map comes from): `scripts/dev/gen_compound_eye.py` — regenerates `compound_eye.npz` from scratch (see Geometry section).
- Integration glue (outside the `vision/` package):
  - `src/flygym/compose/fly/base_fly.py:549-590` — `add_vision()` reads `vision.yaml`, adds one MuJoCo `<camera>` per eye (`l_eye_cam`, `r_eye_cam`) as a child body of the fly's head/eye segment, with `fovy=info["fovy_per_eye"]`, plus a small visualization marker sphere.
  - `src/flygym/simulation.py:408-485` — `Simulation.get_raw_vision()` renders both eye cameras via `mj.Renderer` and applies `Retina.correct_fisheye`; `Simulation.get_ommatidia_readouts()` then calls `Retina.raw_image_to_hex_pxls()` on each rendered frame.
- Tests: `tests/core/test_vision.py` (~470 lines) — pure `Retina` unit tests (no MuJoCo needed) plus `NeuroMechFly.add_vision()` / `Simulation` integration tests; the latter are `skipif(SKIP_RENDERING_TESTS=1)` since they need a headless GL/EGL context.

## Geometry

- **721 ommatidia per eye** (`num_ommatidia_per_eye` in `vision.yaml`), 1442 total for both eyes. This is close to, but not exactly, the "~800/eye" figure used elsewhere in this project — flygym's number comes from a specific synthetic hex-tiling formula, not a real per-eye ommatidia count (see below).
- `fovy_per_eye: 157` degrees — vertical field of view of each eye's MuJoCo perspective camera. Horizontal FOV isn't stored explicitly; MuJoCo derives it from image aspect ratio (450/512 ≈ 0.879), which works out to roughly ~154° horizontal (my own trig from the pinhole-camera formula, not stated anywhere in the code/docs).
- Per a web search summarizing the NeuroMechFly v2 paper (Wang-Chen et al., *Nature Methods* 2024) — not independently re-derived here — the two eyes' **combined field of view is ~270° azimuth with ~17° binocular overlap**, and "721 bins on a hexagonal grid" is the number quoted for the ommatidia count, matching this repo's constant. Treat this FOV/overlap number as paper-level biology context, not something computable purely from `vision.yaml`.
- **No explicit per-ommatidium azimuth/elevation table exists anywhere in the repo.** The "retina mapping" data structure is `ommatidia_id_map`, a **flat pixel-to-ommatidium raster lookup table** in a rectilinear 512x450 image, not an ommatidium-to-3D-angle table. Geometry is baked in implicitly through: the MuJoCo camera's `fovy`/aspect ratio, the `correct_fisheye` warp, and the *pixel-space* hex-tiling algorithm below — none of these expose azimuth/elevation as data you can read off.
- How the map is actually built, `scripts/dev/gen_compound_eye.py:129-176` (`main()`):
  - Lays a regular hexagonal lattice directly in 2D Cartesian pixel-space (`generate_centers`, `hex_vertices`), no sphere or eye surface involved.
  - Crops it to a big hexagonal bounding shape of side length 16 (`RETINA_SIDE_LEN_HEX = 16`), giving `calc_area_in_hexagons(16) = 3*16**2 - 3*16 + 1 = 721` small hexagons — this is where 721 comes from, it's just the closed-form cell count of a hex-of-hexagons with side 16, unrelated to a biological ommatidia count.
  - Rasterizes the hexagon polygons onto a `(512, 450)` grid with `rasterio.features.rasterize` (`rasterize_hex_id_map`, line 112-115), assigning each raster pixel the 1-based index of the polygon (ommatidium) that contains it.
  - Assigns pale/yellow identity with `build_pale_mask` (line 118-126): a flat 30/70 split, **randomly shuffled** with a fixed `RNG_SEED = 0` — not derived from any biological spatial pattern (real fly retinas have known local biases in pale/yellow placement, e.g. dorsal-rim ommatidia are a third type not modeled here at all).
- Eye camera placement (`vision.yaml:sensors`, neuromechfly model):
  ```yaml
  l_eye_cam:
    parent: l_eye
    rel_pos: [-0.03, 0.38, 0]
    orientation: [1.57, 0.00, -0.47]
  r_eye_cam:
    parent: r_eye
    rel_pos: [-0.03, -0.38, 0]
    orientation: [-1.57, 3.14, 0.47]
  ```
  Two independent full-frame MuJoCo perspective cameras, one per eye, each rendering a `(512, 450, 3)` rectilinear RGB image — there is no per-ommatidium raytracing at the MuJoCo level; the hex sampling happens purely as a post-render pixel-binning step (next section).

## Sampling: raw image to per-ommatidium values

Core function, `src/flygym/vision/retina.py:221-235`:
```python
@staticmethod
@nb.njit(parallel=False)
def _raw_image_to_hex_pxls(
    raw_img, ommatidia_id_map, num_pixels_per_ommatidia, pale_type_mask
):
    vals = np.zeros((len(num_pixels_per_ommatidia), 2))
    img_arr_flat = raw_img.reshape((-1, 3))
    hex_id_map_flat = ommatidia_id_map.ravel()
    for i in nb.prange(hex_id_map_flat.size):
        hex_pxl_id = hex_id_map_flat[i] - 1
        if hex_pxl_id != -1:
            hex_pxl_size = num_pixels_per_ommatidia[hex_pxl_id]  # num raw pxls
            ch_idx = pale_type_mask[hex_pxl_id]
            vals[hex_pxl_id, ch_idx] += img_arr_flat[i, ch_idx + 1] / hex_pxl_size
    return vals / 255
```
Public wrapper, `retina.py:111-132` (`raw_image_to_hex_pxls`), just forwards to this Numba kernel with the instance's `ommatidia_id_map`, `num_pixels_per_ommatidia` (precomputed pixel counts per ommatidium id, `retina.py:102-103`), and `pale_type_mask`.

- **Mechanism**: for every raw pixel, look up its ommatidium id from `ommatidia_id_map`; skip if background (`hex_pxl_id == -1`). Accumulate `raw_pixel_value / count_of_pixels_in_that_ommatidium` — i.e. plain **mean pooling** of the raw image over each ommatidium's pixel footprint, no Gaussian/soft weighting.
- **Which raw channel is read**: `ch_idx = pale_type_mask[hex_pxl_id]` is `0` (yellow-type) or `1` (pale-type). The code reads `img_arr_flat[i, ch_idx + 1]`, i.e. **channel index 1 (G) for yellow-type ommatidia, channel index 2 (B) for pale-type ommatidia. The R channel of the rendered image (index 0) is never read anywhere in this function.**
- **Output**: `(num_ommatidia_per_eye, 2)` float array, values in `[0, 1]` (divided by 255). Confirmed by `tests/core/test_vision.py::TestRetinaRawImageToHexPxls::test_output_shape` and `::test_pale_ommatidia_use_green_yellow_use_blue`.
- **What the two output columns mean — this is not an R1-R6-vs-R7/R8 split.** Because `ch_idx` is written to `vals[hex_pxl_id, ch_idx]` and the *other* column is left at `0`, each ommatidium row has **exactly one non-zero entry**: yellow-type ommatidia populate column 0 (sourced from G), pale-type populate column 1 (sourced from B); the unused column just stays `0`. `simulation.py:463-478` documents this explicitly:
  > "the last dimension corresponds to the yellow- and pale-type ommatidia, in that order. Zero values indicate that the ommatidium is of the other type... if `readouts[0, 5, 0]` is 0, it means that the 5th ommatidium is of pale type, and the user should look at `readouts[0, 5, 1]` instead."

  So functionally this is **one scalar brightness-ish reading per ommatidium, routed into one of two type-keyed output slots** — not two independent simultaneous R7/R8-style photoreceptor channels, and not an R1-R6 broadband channel at all.
  - Note: the in-code test docstring (`test_vision.py`, `TestRetinaRawImageToHexPxls::test_pale_ommatidia_use_green_yellow_use_blue`) states in its prose "Pale-type ommatidia store readings in channel 0 (G); yellow-type in channel 1 (B)" — this is **backwards** relative to what the test's own assertions and the `_raw_image_to_hex_pxls` code actually verify (yellow to channel 0/G, pale to channel 1/B). Flagging so this doesn't get copied as fact.

Reverse direction (visualization only, not used for RL input), `retina.py:134-188` / kernel at `retina.py:237-246` (`hex_pxls_to_human_readable` / `_hex_pxls_to_human_readable`): scatters an `(N, ...)` ommatidia reading back onto the `(nrows, ncols, ...)` raster using the same id map, for turning a reading back into a human-viewable hex-blocked image.

`correct_fisheye`, `retina.py:190-219` (wrapper) / `248-276` (kernel `_correct_fisheye`): a generic per-pixel inverse-fisheye remap of a `(nrows, ncols, 3)` image, parameterized by `zoom` and `distortion_coefficient` from `vision.yaml`. It's independent of the ommatidia map and is applied to the raw MuJoCo render *before* `raw_image_to_hex_pxls` (see `simulation.py:452-461`), to compensate for the fact that a rectilinear camera over-represents the periphery of a wide FOV relative to what the compound eye should sample there.

## Colour model

- **No spectral/hyperspectral rendering at all.** MuJoCo renders plain 8-bit RGB; there is no UV channel, no wavelength-dependent material properties, nothing beyond standard RGB.
- Pale/yellow is a **hard boolean identity per ommatidium**, assigned once at data-generation time by a random 30/70 shuffle (`gen_compound_eye.py`, fixed seed), not through any simulated photoreceptor spectral sensitivity curve. The sampler then does a **hard channel selection** (G for yellow, B for pale) rather than a weighted spectral response — a coarse proxy for "pale ommatidia's R7p/R8p are more blue/UV-shifted, yellow's R7y/R8y more green/UV-shifted," implemented as a simple channel pick.
- **R1-R6 (the broadband/achromatic "brightness" photoreceptors) have no representation in flygym's retina model at all.** There is no summed/averaged/achromatic output channel separate from the pale-vs-yellow chromatic pick — this is the biggest structural gap relative to this project's intended "R1-R6 (brightness) + R8 (colour)" per-ommatidium mapping. Reusing flygym's `Retina` as-is would only give you a single yellow/pale-keyed scalar, not two independent brightness+colour values.
- R7 and R8 are not distinguished either — the "pale/yellow" identity in flygym is a single per-ommatidium property, standing in for the R7/R8 inner-photoreceptor chromatic pathway as a whole, not per-cell.

## Dependencies

- `Retina` itself (`retina.py`) only imports `numpy`, `numba` (for the two `@nb.njit` kernels), `pyyaml` (to read `vision.yaml`), and `flygym.assets_dir` (a path constant). **No MuJoCo import in this file.**
- `raw_image_to_hex_pxls`, `hex_pxls_to_human_readable`, and `correct_fisheye` all take/return plain NumPy arrays and have no dependency on a live simulation — **the sampler is usable standalone on any `(H, W, 3)` uint8 image**, as long as you also supply/construct an `ommatidia_id_map` (and `pale_type_mask`, or your own replacement mask) of matching `(H, W)` shape.
- MuJoCo only enters via `Simulation.get_raw_vision()` (`simulation.py:408-461`), which needs a compiled MuJoCo model with `add_vision()`-registered eye cameras and (for real pixels) a headless GL/EGL context. That half of the pipeline is irrelevant if you're feeding in your own rendered frames (e.g. from Craftax).
- It's **NumPy + Numba JIT, not JAX.** `jaxtyping` appears elsewhere in the codebase (e.g. `simulation.py` return-type annotations like `Float[np.ndarray, "..."]`) purely for static type hints on NumPy arrays — it doesn't imply any JAX array/tracing involvement in the vision pipeline.
- License: Apache-2.0 for the repo/module; the `correct_fisheye` algorithm is a Numba port of `Gil-Mor/iFish` (MIT), credited in the docstring.

## Adaptation notes for our case

Our source image is a small top-down Craftax-Classic RGB frame, agent fixed at the center — not a first-person/eye-level camera. Two concrete reuse strategies, listed factually; which to pick is a design decision for the project owner.

### Option (a): split frame into left/right halves as two "eye cameras"

- Crop the rendered Craftax frame down its vertical midline into a left-half and right-half sub-image (optionally with some overlap for a "binocular" region), resize each half to whatever `(nrows, ncols)` you configure, and feed each half into `retina.raw_image_to_hex_pxls` as if it were one eye's `raw_img`.
- Code implications:
  - `Retina(nrows=..., ncols=...)` accepts overridden dimensions (`retina.py:68-99`), but `ommatidia_id_map`/`pale_type_mask` still default to the shipped 512x450/721-ommatidia arrays — if your half-frame resolution differs from 512x450 you must supply a matching `ommatidia_id_map` (own array or a label-preserving nearest-neighbor resize of the shipped one) and a matching `pale_type_mask`, since `_raw_image_to_hex_pxls` indexes directly by pixel position with no bounds/shape check beyond what NumPy raises.
  - `correct_fisheye` assumes a rectilinear MuJoCo camera; there's no obvious equivalent for a top-down game frame — most likely skip it and call `raw_image_to_hex_pxls` directly on the cropped half.
  - No MuJoCo needed at all in this path — bypass `Simulation.get_raw_vision`/`get_ommatidia_readouts` entirely and call the `Retina` methods directly on NumPy crops you produce yourself.
  - Given the colour-model gap above, the pale/yellow `ch_idx`-selection semantics likely need to be replaced by a custom sampler (same mean-pooling-by-id-map pattern as `_raw_image_to_hex_pxls`, but writing R1-R6 and R8 as two independent outputs per ommatidium instead of one value routed by type) rather than reused verbatim.
  - Semantic caveat: "left half of a top-down frame = left eye" is a convenient reuse of the pooling *mechanics*, not a re-derivation of fly optics — a fly's eyes both look outward+forward across most of a combined ~270° field, they don't split a scene down the middle. "Azimuth" in this scheme just becomes left-right pixel position within a half-frame, a different quantity than a true azimuthal angle around the fly's head.

### Option (b): radial sampling centred on the agent

- Craftax is already top-down with the agent centered, so a polar reparametrization is a closer geometric match: ommatidium azimuth maps to angle-around-the-agent in the frame, elevation maps to radial distance from the agent (e.g. near rings = one elevation band, far rings = another).
- Code implications:
  - None of `ommatidia_id_map`/`correct_fisheye`/MuJoCo rendering is reusable as-is, since they're built around a rectilinear pinhole image. You'd write a new sampling function, structurally similar to `_raw_image_to_hex_pxls` (`retina.py:221-235`) but computing each ommatidium's source pixel from its assigned `(azimuth, elevation)` directly — e.g. `row = cy + r*sin(theta)`, `col = cx + r*cos(theta)` with `r = f(elevation)`, then point-sample or mean-pool a small neighborhood/annulus-wedge around that point.
  - You'd need your own per-ommatidium `(azimuth, elevation)` table (flygym doesn't provide one at all, per the Geometry section) — e.g. N angular positions per ring x M rings, generated directly in polar coordinates rather than flygym's flat Cartesian hex tiling (`gen_compound_eye.py`).
  - The `ommatidia_id_map`-plus-`raw_image_to_hex_pxls` *mechanism* could still be reused if you rasterize your own polar hex lattice into an id-map image the same shape as the Craftax frame (same rasterize-by-polygon approach as `gen_compound_eye.py`, just with hexagon centers placed on a polar grid) — `raw_image_to_hex_pxls` then works unmodified once you also swap in a mask/channel-selection scheme matching R1-R6/R8 semantics instead of pale/yellow.
  - `correct_fisheye`, MuJoCo, and `Simulation.get_raw_vision`/`get_ommatidia_readouts` are all irrelevant to this path.
  - "Binocular overlap" doesn't really apply to a single monocular top-down frame; two "eyes" here would more likely be two full radial samplings with different angular offsets/weightings than a true left/right split.

## Mapping to connectome neurons

- Checked directly against this project's own downloaded MaleCNS annotation table, `data/body-annotations.feather` (211,577 rows, MaleCNS v1.0):
  - **`assignedOlHex1` and `assignedOlHex2` are real columns in this table** — float-valued (integer-like), ranges roughly `1-36` (hex1) and `1-39` (hex2), i.e. an oblique two-coordinate hex-lattice index for retinotopic columns.
  - They are populated **only for `superclass == "ol_intrinsic"`** cell types — classic lamina/medulla columnar interneurons: `L1, L2, L3, L5, C2, C3, T1, Mi1, Mi4, Mi9, Tm1, Tm2, Tm4, Tm9, Tm20, ...`. Total rows with a non-null `assignedOlHex1`: 23,720.
  - Per-type, per-side counts are close to one cell per retinotopic column: e.g. `L1` has 875 (left) / 892 (right) cells, essentially all with unique `(assignedOlHex1, assignedOlHex2)` pairs — i.e. ~875-892 columns per eye in this dataset, which lines up well with this project's "~800 ommatidia/eye" figure.
  - **Photoreceptor rows themselves have `assignedOlHex1`/`assignedOlHex2` = `NaN`, with zero exceptions.** Checked directly: all 6,091 rows with `type` in `{R1-R6, R7p, R7y, R7d, R7_unclear, R7R8_unclear, R8p, R8y, R8d, R8_unclear}` (`superclass == "ol_sensory"`) have null hex coordinates. So **the hex-column coordinate is not available directly on the photoreceptor rows** in this table — R1-R6 is even stored as a single lumped `type` label per body (not R1, R2, ... R6 individually), and there's no `receptorType`/opsin field populated for any of them either (checked; always `None`).
  - Practical implication: to place a photoreceptor body in the hex grid, you'd need to go through **connectivity**, not the annotation row directly — e.g. find each photoreceptor's postsynaptic column-mate (an `L1`/`L2`/`Mi1`/etc. body in the lamina that *does* carry `assignedOlHex1/2`) via the synapse/edge table (`connectome-weights-male-cns-v1.0-minconf-0.5.feather`, listed in `tracker.md` but not present in this local `data/` yet — only `body-annotations.feather` and `body-neurotransmitters.feather` are downloaded), and inherit that column's coordinate. This wasn't executed here; flagging as the next concrete step rather than doing it in this pass.
- Cross-checked against the FlyWire optic-lobe connectome literature (web search + one successful `WebFetch` of the Nern et al. bioRxiv preprint, `10.1101/2024.04.16.589741v2`, since the published Nature version is paywalled):
  - "The lenses of the fly eye form a hexagonal grid and are mapped onto a hexagonal coordinate system in the medulla... The coordinates correspond one-for-one to lenses on the compound eye (except for some lenses on the edge)." — i.e. the same style of oblique 2-coordinate hex indexing used by `assignedOlHex1/2` above is the field's standard approach, not something specific to MaleCNS.
  - Pale/yellow identity in the real connectome is **not** a geometric/imaging property — it's inferred from connectivity: cluster R7/R8 by downstream synaptic partner type (`Tm5a` vs `Tm5b`, `Dm8a` vs `Dm8b`), corroborated by co-localization/connectivity with cell type `aMe12`. This is a materially different (and biologically grounded) source for "pale/yellow, i.e. chromatic identity" than flygym's random 30/70 shuffle, and — per the `data/body-annotations.feather` check above — MaleCNS's own R7/R8 rows already carry that call directly as `type in {R7p, R7y, R7d, R8p, R8y, R8d}` (`d` = dorsal-rim-area type, a third category flygym's model doesn't have at all).
  - FlyWire v783 counts (from a WebSearch summary of Nern et al., not independently re-verified): R7 and R8 ~650 cells each, R1-R6 ~3,400 total, for one hemisphere/optic lobe, with ~800 ommatidia/columns per eye overall — broadly consistent with the MaleCNS per-side column counts found above.
- Not independently verified: I could not find the literal strings `assignedOlHex1`/`assignedOlHex2` documented in any public neuPrint/Clio schema page, the `flyconnectome/ol_annotations` GitHub repo, or the `natverse/malecns` package docs via web search — but they are confirmed to exist as real, populated columns in this project's own already-downloaded `data/body-annotations.feather`, which is the authoritative source to use directly rather than relying on external docs.

## Open questions for spec

- flygym's `Retina` only ever emits **one** scalar per ommatidium (routed into a yellow-or-pale output slot); the project wants **two** independent values per ommatidium (R1-R6 brightness + R8 colour). Confirms we can't reuse `raw_image_to_hex_pxls` verbatim — do we want a custom sampler that reuses just its mean-pooling-by-id-map pattern, computing both an achromatic (e.g. average of RGB or a fixed weighting) and a chromatic value per ommatidium?
- Frame-splitting strategy: half-frame-per-eye (option a, mechanically closer to flygym's existing pipeline) vs. agent-centred radial/polar sampling (option b, geometrically closer to what a compound eye actually does for a creature meant to see in a ring around itself, but needs a from-scratch ommatidium lattice generator)? These are mutually exclusive.
- Do we source per-ommatidium pale/yellow (or general R7/R8 chromatic) identity from real MaleCNS connectivity-derived type calls (`R7p`/`R7y`/`R8p`/`R8y`/... in `data/body-annotations.feather`, joined to hex columns via connectivity — not yet computed here), or keep a synthetic ratio like flygym's default (30/70 random, fixed seed) for simplicity?
- Do we need any fisheye/lens-distortion correction at all for a top-down, roughly-orthographic Craftax camera, or is `correct_fisheye` purely an artifact of flygym's rectilinear-MuJoCo-camera setup that should be dropped entirely?
- Photoreceptor-to-hex-column linking needs the actual synapse edge table (`connectome-weights-male-cns-v1.0-minconf-0.5.feather`, referenced in `tracker.md` but not yet present in `data/`) to find each photoreceptor's postsynaptic column-mate and inherit its `assignedOlHex1/2` — worth downloading that file and doing this join before finalizing the ommatidium-index -> connectome-body-ID lookup.
- Do we need two separate "eyes" for a single top-down monocular frame at all, or is one shared ~800-1600-unit retina acceptable given Craftax has no true binocular geometry?
