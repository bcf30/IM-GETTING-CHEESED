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

## Design hypotheses

| ID | Outcome |
|---|---|
| H1 | Pending (data inspection task). |
| H2 | Pending. |
| H3 | Pending. |
| H4 | Not falsified in a first check: the xarray route gave an ImagePlus with 6 slices, 1 channel, 1 frame, 8-bit for a (6, 8, 10) mask, and the mask round-trip was exact. `tests/test_fiji.py` will check it formally. |
| H5 | **Supported** for the `direct` backend: MorphoLibJ labelling, size opening and region analysis run headless through scyjava calls. The `plugin` backend fails at Analyze Regions 3D. The 3D ImageJ Suite was not needed. No scikit-image code is used. |
| H6 | Probe of two 3³ cubes touching only at a corner: 1 component at 26-connectivity, 2 at 6-connectivity. This is the defined behaviour of 26-connectivity, not evidence that it merges clearly separate fat channels, so `CONNECTIVITY_3D` stays 26. The visual check on real volumes is still open. |
| H7 | A headless 3D max Feret exists (`MaxFeretDiameter3D`), so the ellipsoid proxy is not needed. Radius-6 and radius-7 digital spheres gave 13.67 and 15.59 voxels. The ±15% comparison against analytic values is left to the measure tests. |
| H8 | Pending (median seconds per volume from the full run). |
| H9 | Pending. |
| H10 | Pending. |
| H11 | Pending. |
| H12 | **Supported on Python 3.14.10.** `imagej.init` succeeded (about 10 s after the first download), and a float32 (5, 7, 9) array survived `to_java_img` → `from_java_array` exactly, with Java dimensions (9, 7, 5). The Python 3.13 fallback was not needed. |
