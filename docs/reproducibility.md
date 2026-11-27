# Reproducibility record

This file records the exact software stack, the Fiji discovery results that the constants in
`mozzavid/fiji.py` were copied from, and the outcome of each design hypothesis. Re-run
`python scripts/discover_fiji.py` after any Fiji, MorphoLibJ or Java change and update both
this file and the constants. The raw discovery output is in `docs/fiji_discovery.json`.

## Environment

Created with `conda env create -f environment.yml` (conda 26.9.4, Miniforge, Windows 11 x64).
The solve succeeded with every pin as written. Exact solved versions:

| Package | Version | Build |
|---|---|---|
| python | 3.14.10 | h53f6dd8_100_cp314 |
| openjdk | 11.0.33.1 (Zulu 11.92+201) | h18f690b_0 |
| maven | 3.10.2 | h36c15f3_0 |
| pyimagej | 1.8.0 | pyhd8ed1ab_0 |
| scyjava | 1.12.5 | pyhd8ed1ab_0 |
| jpype1 | 1.7.1 | py314h13fbf68_1 |
| imglyb | 2.1.0 | pyhecae5ae_1 |
| jgo (transitive) | 3.3.3 | pyhd8ed1ab_0 |
| xarray (transitive) | 2026.9.0 | pyh5ded981_0 |
| numpy | 2.5.5 | py314h02f10f6_1 |
| scipy | 1.18.1 | py314h8c99b21_4 |
| pandas | 3.1.1 | np2py314h1b5b07a_0 |
| matplotlib | 3.11.2 | py314h2061cd4_1 (matplotlib-base) |
| tifffile | 2026.11.16 | pyhc364b38_0 |
| pytest | 9.2.0 | pyhc364b38_1 |

Runtime versions reported by `fiji.runtime_versions()`:

| Component | Version |
|---|---|
| Fiji / ImageJ 1.x | 2.17.0 / 1.54p |
| Java (`java.version`) | 11.0.33.1, `java.home` = the env's `Library\lib\jvm` |
| PyImageJ | 1.8.0 |
| scyjava | 1.12.5 |

### Java selection

scyjava 1.12.5 defaults to `fetch="always"`, which downloads its own Zulu JRE 11 through
cjdk and ignores the conda openjdk. The first discovery run did exactly that. `fiji._start_imagej`
now calls `scyjava.config.set_java_constraints(fetch="auto")`, so the JVM is the conda-pinned
openjdk found through `JAVA_HOME`. `fetch="never"` is not usable: pyimagej 1.8.0's
`_guess_java_version` raises `UnboundLocalError` (`major_version`) when no JVM is running and
no fetch is planned.

### MKL DLL load on Windows (unactivated interpreter)

**Symptom.** Calling the env's interpreter by absolute path without `conda activate`
(for example `C:\Users\...\envs\mozzavid\python.exe script.py`) crashed with a Windows fatal
exception `0xC06D007F` (`ERROR_DELAY_LOAD_FAILED`) inside any LAPACK call: `numpy.linalg.inv`
on a 2x2 identity, `numpy.linalg.svd`, and therefore `matplotlib`'s `tight_layout`
(which inverts a transform during tick/title layout). Running the same code through
`conda run -n mozzavid python ...` or in an activated shell worked.

**Cause.** MKL (the BLAS/LAPACK backend, `mkl 2026.2.1`) lives only in the env's `Library\bin`.
Its runtime loads `mkl_rt.3.dll` and then *delay-loads* `mkl_intel_thread.3.dll` ->
`libiomp5md.dll`. The delay-load search needs `Library\bin` on the process `PATH`. Conda
activation prepends `Library\bin` to `PATH`; a bare interpreter launch does not, so the
delay-load fails. Each DLL loads fine on its own by explicit path, and
`os.add_dll_directory(Library\bin)` alone does **not** fix it: MKL's lazy secondary load
uses the classic `PATH` search, not the `add_dll_directory` set.

**Fix (committed to the env).** A startup `.pth` prepends the env's `Library\bin` to `PATH`
at interpreter start, mirroring activation:

```
# <env>\Lib\site-packages\_mozzavid_mkl_dlldir.pth  (single physical line)
import os, sys; _d = os.path.join(sys.prefix, 'Library', 'bin'); sys.platform == 'win32' and os.path.isdir(_d) and (os.add_dll_directory(_d), os.environ.__setitem__('PATH', _d + os.pathsep + os.environ.get('PATH', '')))
```

A `.pth` line beginning with `import` runs before any package import, so MKL resolves on the
first LAPACK call. With this file present the bare interpreter runs the full test suite
(`50 passed`) and renders figures without `conda run`. The file lives inside the environment,
not in the repository; recreating the env from `environment.yml` drops it, so re-add it or run
through activation. This is an environment/runtime fix only; no pipeline code depends on it.

## Fiji coordinate

```
FIJI_COORDINATE = "sc.fiji:fiji:2.17.0"
EXTRA_ENDPOINTS = ["fr.inra.ijpb:MorphoLibJ_:1.6.3"]
```

- `sc.fiji:fiji:2.18.0` is the latest release on maven.scijava.org, but it does not start on
  Java 11: `sc.fiji.compat.DefaultFijiService` has class file version 65 (Java 21) and the jgo
  lock file reports `min_java_version = 21`. 2.17.0 (parent pom-scijava 42.0.0, base 21.1.0,
  release target Java 8) is the newest release that runs on the pinned openjdk 11.
- MorphoLibJ is not a dependency of `sc.fiji:fiji` (discovery: all three `inra.ijpb.*` classes
  not importable, no MorphoLibJ commands registered). Requirement 6.14 therefore applies.
  `sc.fiji:MorphoLibJ_` does not exist on maven.scijava.org; the artifact is
  `fr.inra.ijpb:MorphoLibJ_`. Version 1.6.3 is the one managed by pom-scijava 42.0.0.
- The 3D ImageJ Suite (`mcib3d`) is not in `sc.fiji:fiji` either. It was not needed (see H5).
- The first run downloads about 680 jars into `~/.jgo` and `~/.m2` (about 15 minutes on the
  development machine). Later starts take about 10 s and need no network.

## Discovered command strings and classes

| Constant | Value | Discovery result |
|---|---|---|
| `MORPHOLIBJ_BACKEND` | `"direct"` | only backend whose label, size-opening and region-analysis steps all ran headless |
| `CMD_CONNECTED_COMPONENTS` | `"Connected Components Labeling"` | registered IJ1 command |
| `CMD_SIZE_OPENING` | `"Label Size Opening"` | registered IJ1 command |
| `CMD_ANALYZE_REGIONS_3D` | `"Analyze Regions 3D"` | registered IJ1 command |
| `CLASS_BINARY_IMAGES` | `inra.ijpb.binary.BinaryImages` | importable; `componentsLabeling(ImagePlus, int, int)` |
| `CLASS_LABEL_IMAGES` | `inra.ijpb.label.LabelImages` | importable; `sizeOpening(ImagePlus, int)` |
| `CLASS_ANALYZE_REGIONS_3D` | `inra.ijpb.measure.region3d.MorphometricFeatures3D` | importable; `computeTable(ImagePlus)` returns `ij.measure.ResultsTable` |
| `CLASS_MAX_FERET_3D` | `inra.ijpb.measure.region3d.MaxFeretDiameter3D` | importable; `computeTable(ImagePlus)` returns `ij.measure.ResultsTable` |

The design's candidate `inra.ijpb.plugins.AnalyzeRegions3D` is importable, but in MorphoLibJ
1.6.3 its only public method is `run(String)`. `MorphometricFeatures3D` is the class the plugin
delegates to, so it replaced the candidate in `CLASS_ANALYZE_REGIONS_3D`.

Backend probe on the 16³ two-sphere image:

| Backend | Labelling | Size opening | Analyze Regions 3D |
|---|---|---|---|
| `plugin` (`ij.py.run_plugin`) | OK, 2 components, 16-bit | OK | FAIL: no `<title>-morpho` ResultsTable is reachable headless |
| `direct` (scyjava calls) | OK, 2 components | OK | OK, `ij.measure.ResultsTable`, 2 rows |
| `mcib3d` | not probed | | |

Settings used by `fiji.py`:

- Labels are 32-bit float (`LABEL_BIT_DEPTH = 32`). Floats represent integers exactly up to
  2^24, more than the 192³ voxels of a volume, so a noisy mask cannot overflow the label range
  as it could with 16-bit labels (65 535).
- `MorphometricFeatures3D` features requested: `VOXEL_COUNT, VOLUME, SURFACE_AREA, SPHERICITY,
  BOUNDING_BOX, CENTROID, EQUIVALENT_ELLIPSOID`. Defaults kept: surface area by Crofton with
  13 directions, Euler connectivity 6 (the Euler number is not requested).
- `size_opening` keeps the original label values; it does not relabel to 1..n.

## COLUMN_MAP

Columns of the direct backend's table (pixel units, uncalibrated input) and their internal
names. `MaxFeret.Diameter` is appended by `fiji.analyze_regions_3d` from `MaxFeretDiameter3D`,
after checking that both tables list the same labels in the same order.

| Fiji column | Internal raw name |
|---|---|
| `Label` | `label` |
| `VoxelCount` | `voxel_count` |
| `Volume` | `volume_px3` |
| `SurfaceArea` | `surface_area_px2` |
| `Sphericity` | `sphericity_morpholibj` |
| `Box.X.Min`, `Box.X.Max`, `Box.Y.Min`, `Box.Y.Max`, `Box.Z.Min`, `Box.Z.Max` | `bbox_x_min_px` … `bbox_z_max_px` |
| `Centroid.X`, `Centroid.Y`, `Centroid.Z` | `centroid_x_px`, `centroid_y_px`, `centroid_z_px` |
| `Elli.Radius1`, `Elli.Radius2`, `Elli.Radius3` | `ellipsoid_radius1_px` … `ellipsoid_radius3_px` |
| `MaxFeret.Diameter` | `feret_max_px` |

Unmapped columns kept under their Fiji names: `Elli.Center.X/Y/Z`, `Elli.Azim`, `Elli.Elev`,
`Elli.Roll`.

Coordinate conventions measured on a radius-6 sphere centred on voxel index 10 in a 32³ volume:
`Centroid.*` = 10.5 (voxel *i* is treated as the cell [*i*, *i*+1], so centroid = index + 0.5);
`Box.*.Min` = 4 and `Box.*.Max` = 17 (minimum index and maximum index + 1). The border flag in
`measure.py` must use these conventions: a face at index 0 has `Min == 0`, a face at index 191
has `Max == 192`.

## Sphericity mapping (MorphoLibJ → Wadell)

MorphoLibJ reports `Sphericity = 36·π·V² / S³`. The pipeline reports Wadell's sphericity
`Ψ = π^(1/3)·(6V)^(2/3) / S`, which equals the cube root of MorphoLibJ's value:

```
Ψ_Wadell = Sphericity_MorphoLibJ ** (1/3)
```

Checked on two digital spheres (radius 6 and 7 voxels): Wadell 1.0076 and 1.0121, whose cubes
1.0231 and 1.0367 match MorphoLibJ's `Sphericity` column to 4 decimals. Both are slightly
above 1 because the Crofton surface estimate is low for small digital objects; this is the
reason for the `≤ 1.05` validation tolerance.

## Committed example run

The committed outputs in `results/` and the five figures in `figures/` come from:

```powershell
python scripts/run_pipeline.py --data-root ./Small --output ./results --figures ./figures --limit 25 --synthetic-samples 4
```

`--limit 25` takes one volume per cheese type (round-robin), the quickest configuration that
still exercises all 25 types and the full output path. Measured on the development machine:
25 volumes, **0 skipped**, per-volume wall time median **≈ 2-4 s** (min 1.3 s, max ~10 s),
object counts 492-2016 per volume, whole run ~90 s plus the synthetic benchmark. The per-volume
cost is well under the design's 5-17 s estimate once the ellipsoid Feret proxy replaced the
unscalable `MaxFeretDiameter3D` (see H7).

A full-dataset run was attempted but did not complete: the JVM exhausted its 4 GB heap after
142 volumes, and a `--limit 120` retry crashed silently at volume 58 (see H8 for the cause and
the batching/heap mitigations). The committed example is therefore the reliable `--limit 25`
run, which is deterministic and small.

Because each cheese type has a single sample at `--limit 25`, the group comparisons and ICC(1)
read "insufficient data" in `anova_results.txt` and `summary.md` by design (both need >= 2
samples per type). A run that includes repeated samples per type (e.g. the full dataset)
populates the statistics; the machinery is verified by `tests/test_stats.py` and by the
`integration`/`smoke` tests on real data.

## Design hypotheses

| ID | Outcome |
|---|---|
| H1 | **Not refuted; still TODO-VERIFY.** `VOXEL_SIZE_UM = 4.4` (0.55 µm raw × 8 Small-downscale) is kept. The committed run's interior-object equivalent diameters (number-weighted medians ~27-30 µm per sample) are plausible for mozzarella fat/serum structure at this resolution and do not conflict with it, but the value is derived from two paper statements, not stated for Small, so it stays marked TODO-VERIFY in `mozzavid/io.py`. All relative results are independent of it. |
| H2 | **`DEFAULT_INVERT = True`.** Montage `figures/polarity_check.png`, central Z slices of 5 volumes spread over `cheese_index`: `cheese_1/1/0.tiff`, `cheese_7/1/0.tiff`, `cheese_13/1/0.tiff`, `cheese_19/1/0.tiff`, `cheese_25/1/0.tiff`. In 4 of the 5 panels (indices 1, 7, 13, 19) the rounded and elongated fat inclusions are **dark** on a **brighter grey matrix**, so Otsu/Li with "True above threshold" selects the matrix; the segmentation must complement it to keep the inclusions, hence `invert=True`. (Panel `cheese_25` shows bright specks on a darker noisy field, a different-looking acquisition, but the dominant and consistent signal is dark-on-bright.) `pipeline.DEFAULT_INVERT` is set from this record in task 13.1. |
| H3 | **Not refuted on the committed run.** Global Otsu on a sigma = 1.5-voxel Gaussian gave segmentations whose overlays (`figures/segmentation_examples.png`) follow the visible dark inclusions, with per-sample foreground fractions in a plausible 0.04-0.33 range across cheese types. No systematic mis-segmentation was seen; `--method li` remains available as the documented fallback if a future check shows per-volume Otsu thresholds drifting > 0.1 within a sample. |
| H4 | Not falsified in a first check: the xarray route gave an ImagePlus with 6 slices, 1 channel, 1 frame, 8-bit for a (6, 8, 10) mask, and the mask round-trip was exact. `tests/test_fiji.py` will check it formally. |
| H5 | **Supported** for the `direct` backend: MorphoLibJ labelling, size opening and region analysis run headless through scyjava calls. The `plugin` backend fails at Analyze Regions 3D. The 3D ImageJ Suite was not needed. No scikit-image code is used. |
| H6 | Probe of two 3³ cubes touching only at a corner: 1 component at 26-connectivity, 2 at 6-connectivity. This is the defined behaviour of 26-connectivity, not evidence that it merges clearly separate fat channels, so `CONNECTIVITY_3D` stays 26. The visual check on real volumes is still open. |
| H7 | **Falsified on scalability; ellipsoid proxy adopted (Requirement 11.14/11.15).** MorphoLibJ's headless `MaxFeretDiameter3D` is accurate on small objects but computes the max pairwise point distance per label at roughly O(points²), so `computeTable` does **not finish** on a real volume (~0.18 foreground fraction, ~1800 components): all other region features together take 0.5–0.6 s, while max Feret ran > 3 min and was killed. `measure._feret_um` therefore uses `2 × largest equivalent-ellipsoid semi-axis` (already in the region table, zero extra cost). Accuracy vs analytic max Feret: sphere r=6 analytic 12 → proxy 12.16 (1.3%) vs MorphoLibJ 13.67 (14%); r=8 → 15.96 (0.3%) vs 17.29 (8%); r=10 → 20.01 (0.1%) vs 21.61 (8%); ellipsoid (12,8,6) analytic 24 → 24.11 (0.5%) vs 25.04; ellipsoid (14,6,6) analytic 28 → 28.02 (0.1%) vs 29.03. The proxy is **more accurate** than MorphoLibJ's discrete corner-to-corner estimate and within the 15% criterion, so the column is kept (not dropped). `fiji.analyze_regions_3d` no longer calls `MaxFeretDiameter3D`; the class stays importable and verified for reference. `tests/test_measure.py::test_feret_within_15_percent_of_analytic` checks the proxy against analytic spheres. |
| H8 | **Supported: the full 591-volume run completes.** The earlier instability was a real memory leak in `segment.cleanup`, not ImageJ1 global state: the two intermediate 32-bit label `ImagePlus` objects from `label_components` and `size_opening` (each a full 192³ image, ~28 MB) were never closed, so the heap grew ~27.4 MB per volume and filled `-Xmx4g` at ~142 volumes (confirmed by per-stage heap attribution: `cleanup` added +27.4 MB/volume, every other stage was flat). `System.gc()` could not reclaim them because `cleanup` held hard references. The fix closes both label images (and the `imp`) in `cleanup`'s `finally`, and `measure.extract` likewise closes its label image; after the fix `cleanup` adds +0.2 MB/volume and the heap is flat. A clean full run (`run_pipeline.py --data-root ./Small --synthetic-samples 20`) then processed **all 591 volumes, 0 skipped**, in **35.5 min of processing (median 3.6 s/volume)**, inside the design's 50 min – 2.8 h estimate. The committed `results/` and `figures/` are from this full run (per-object table > 50 MB, so written as `per_object_metrics.csv.gz`). `--limit 150` remains the documented "< 1 h" path. |
| H9 | **Not executed headless in this environment; documented path kept.** The conda/PyImageJ install resolves Fiji through Maven and has no standalone `ImageJ`/`Fiji` GUI launcher on PATH, so `ImageJ --headless -macro` and the GUI `open()` float16 check could not be run here. `macros/batch_particle_analysis.ijm` is written against the command strings pinned in `mozzavid/fiji.py` (`Connected Components Labeling`, `Label Size Opening`, `Analyze Regions 3D`, connectivity 26) and reads the float32 copies exported by `scripts/validate_segmentation.py --export-float32`, which is the safe default because ImageJ 1.x has no half-float type. The macro-vs-Python agreement check (foreground fraction ≤ 0.01, object count ≤ 5%) and the float16-open test remain to be run by a reviewer with a Fiji GUI/launcher; the procedure is in the macro header and the README. |
| H10 | **Measured on the committed `--limit 25` run.** `stats.icc1` is wired into `summary.md` per Stat_Metric and reported for every full run. At `--limit 25` each cheese type has a single sample, so ICC(1) reads "insufficient data" by design (it needs >= 2 samples with >= 2 volumes each); a run with repeated samples per type populates it. The sample is kept as the statistical unit regardless, the conservative choice (design H10). |
| H11 | **Falsified on the full 591-volume run; interpretation note added, schema unchanged.** 41.1% of volumes (243/591, well above the 20% threshold) have a largest object holding > 50% of the foreground voxels (median ratio 0.41). At 4.4 µm/voxel many fat domains percolate into one large connected object, so the size metrics are partly "connected-domain" sizes, not single-globule sizes. Per the design, adding the `largest_object_fraction` column is **optional** when H11 is falsified; it is left out to keep Requirement 14.2's schema, and this caveat is recorded here instead. The `touches_border` flag and interior-only size statistics already limit the bias from these large domains, and `foreground_fraction`/`objects_per_mm3` (which do not depend on per-object size) remain the most robust discriminators (they also give the strongest group separation: p ≈ 1e-11 and 1e-16). |
| H12 | **Supported on Python 3.14.10.** `imagej.init` succeeded (about 10 s after the first download), and a float32 (5, 7, 9) array survived `to_java_img` → `from_java_array` exactly, with Java dimensions (9, 7, 5). The Python 3.13 fallback was not needed. |
