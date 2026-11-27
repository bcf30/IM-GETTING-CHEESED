# mozzavid-microstructure

A reproducible pipeline for **food-microstructure analysis** of 3D X-ray CT volumes. It
segments and quantifies the microstructure of mozzarella cheese from the
[MozzaVID](https://arxiv.org/abs/2412.04880) **Small** split (591 volumes, 25 cheese types,
149 physical samples). Every image-processing step (Gaussian denoising, Otsu/Li thresholding,
small-object removal, connected-component labelling, 3D region measurement) runs inside
**Fiji/ImageJ headless** through [PyImageJ](https://github.com/imagej/pyimagej). Python handles
orchestration, I/O, per-sample aggregation, statistics and plotting; scikit-image is not used.

The pipeline aggregates per-object measurements per physical cheese sample (the independent
statistical unit), compares the 25 cheese types with one-way ANOVA or Kruskal–Wallis plus the
matching post-hoc test, and renders five figures. A synthetic sphere-packing extension validates
the whole measurement chain against ground truth without needing the dataset.

## Setup

Miniforge/conda is required (it provides Java 11 and Maven alongside Python). From a new shell:

```pwsh
conda env create -f environment.yml
conda activate mozzavid
```

The first call into Fiji starts the JVM, which takes **15–20 s**, and on the very first run it
downloads Fiji and MorphoLibJ from Maven (about 680 jars, ~15 min, one time only). After that the
JVM starts in ~10 s and no network is needed.

## Test

```pwsh
pytest -q                          # unit tests; run Fiji on synthetic data, skip data-dependent tests
pytest -q -m "integration or smoke"   # data-dependent tests; need ./Small
```

Unit tests run without the dataset. Integration and smoke tests self-skip with a message naming
the resolved data root when `./Small` is absent.

## Run

```pwsh
# Stratified subset expected to finish in under 1 hour (6 volumes per cheese type):
python scripts/run_pipeline.py --data-root ./Small --output ./results --figures ./figures --limit 150

# Full 591-volume run — see the note below; long runs need a larger JVM heap or batching:
python scripts/run_pipeline.py --data-root ./Small --output ./results --figures ./figures --synthetic-samples 20
```

> **Long runs are unstable on this Fiji stack.** Per-volume processing is fast (~2-4 s), but a
> full sequential run exhausted the 4 GB JVM heap after ~140 volumes in one attempt and crashed
> at ~60 in another (ImageJ1 legacy global state retains per-volume images). For the full
> dataset, raise `JVM_MAX_HEAP` in `mozzavid/fiji.py` and/or process in batches (restart the
> process per batch). The committed example in `results/` uses `--limit 25` (~90 s,
> deterministic). See H8 in `docs/reproducibility.md`.

Outputs land in `./results` (`per_object_metrics.csv`, `per_sample_summary.csv`,
`anova_results.txt`, `summary.md`, `skipped_volumes.txt`) and `./figures` (five PNGs). The data
root can also be set with the `MOZZAVID_DATA_ROOT` environment variable; see `data/README.md`.

Useful flags: `--method {otsu,li}`, `--sigma` (voxels, > 0), `--min-size` (voxels),
`--invert/--no-invert` (default `True`: the fat inclusions are the darker phase), `--limit N`
(stratified subset), `--synthetic-samples N` (≥ 2), `--log-level`.

Validate the segmentation on a few volumes and export float32 copies for the macro:

```pwsh
python scripts/validate_segmentation.py --data-root ./Small --n 5 --export-float32 ./exports
```

## Fiji validation macro

`macros/batch_particle_analysis.ijm` mirrors the Python chain in pure ImageJ so the segmentation
can be reproduced inside Fiji. ImageJ 1.x has no half-float type, so the macro reads the float32
copies written by `validate_segmentation.py --export-float32`.

- **GUI:** Fiji → *Plugins › Macros › Run…*, pick the macro, and choose the exported-TIFF folder
  and an output folder when prompted.
- **Headless:** `ImageJ --headless -macro macros/batch_particle_analysis.ijm "inputDir,outputDir,1.5,50,true"`

The two runs agree when the foreground fraction differs by at most 0.01 absolute and the object
count by at most 5%; exact equality is not expected because IJ1 kernel truncation and histogram
binning differ slightly from ImageJ Ops.

## Limitations

- **`VOXEL_SIZE_UM = 4.4` is a derived value pending verification.** It is 0.55 µm (the paper's
  raw pixel size) × 8 (the Small downscale factor), inferred from two statements in Pieta et al.,
  not stated explicitly for Small. All relative results (fractions, rank tests, synthetic
  discrimination) are independent of it; only absolute µm/mm³ values scale with it. Confirm
  against the dataset card or the authors before publishing absolute units.
- **The Small split (~4.4 µm voxels) cannot resolve fat globules of about 3 µm.** The segmented
  phase is reported as `foreground_fraction` (fat plus unresolved voids), not porosity.
- **Post-hoc power is limited:** 25 cheese types give 300 pairwise comparisons.

## Citation

MozzaVID dataset: S. B. Pieta et al., *MozzaVID: Mozzarella Volumetric Image Dataset*,
arXiv:2412.04880. Please cite the dataset authors when using the data.

## License

MIT. See [`LICENSE`](LICENSE).
