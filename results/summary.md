# MozzaVID microstructure summary

- Data root: `C:\Users\Ramiro\Documents\VSCode\IM-GETTING-CHEESED\Small`
- Samples summarised: 25
- Voxel size: 4.4 µm (derived, TODO-VERIFY)
- Gaussian sigma: 1.5 voxels (6.6 µm)
- Min object size: 50 voxels (4259 µm³)
- Threshold: otsu, invert=True

## Group comparison per metric

### foreground_fraction

- Comparison: insufficient data
- ICC(1): insufficient data

### objects_per_mm3

- Comparison: insufficient data
- ICC(1): insufficient data

### median_equiv_diameter_um

- Comparison: insufficient data
- ICC(1): insufficient data

### d50_um

- Comparison: insufficient data
- ICC(1): insufficient data


## Skipped volumes

None.

## Limitations

- With 25 cheese types, the post-hoc stage tests 300 pairs; power per pair is limited at 5–6 samples per type, so only large differences are detectable.
- The Small split resolves structures down to ~3 µm; globules below this are not separable at the derived 4.4 µm voxel size.
- `foreground_fraction` is the volume fraction of the segmented (mostly fat) phase, not porosity.
