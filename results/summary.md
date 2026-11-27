# MozzaVID microstructure summary

- Data root: `C:\Users\Ramiro\Documents\VSCode\IM-GETTING-CHEESED\Small`
- Samples summarised: 149
- Voxel size: 4.4 µm (derived, TODO-VERIFY)
- Gaussian sigma: 1.5 voxels (6.6 µm)
- Min object size: 50 voxels (4259 µm³)
- Threshold: otsu, invert=True

## Group comparison per metric

### foreground_fraction

- Test: Kruskal-Wallis
- Statistic: 97.02, p = 9.648e-11
- epsilon_squared: 0.6555
- Levene p: 5.116e-08, groups: 25
- ICC(1) of volumes within samples: 0.7254

### objects_per_mm3

- Test: Kruskal-Wallis
- Statistic: 129.8, p = 1.658e-16
- epsilon_squared: 0.8773
- Levene p: 3.681e-10, groups: 25
- ICC(1) of volumes within samples: 0.8097

### median_equiv_diameter_um

- Test: Kruskal-Wallis
- Statistic: 132, p = 6.725e-17
- epsilon_squared: 0.8919
- Levene p: 1.621e-07, groups: 25
- ICC(1) of volumes within samples: 0.2763

### d50_um

- Test: Kruskal-Wallis
- Statistic: 133, p = 4.379e-17
- epsilon_squared: 0.8988
- Levene p: 0.01657, groups: 25
- ICC(1) of volumes within samples: 0.4034


## Skipped volumes

None.

## Limitations

- With 25 cheese types, the post-hoc stage tests 300 pairs; power per pair is limited at 5–6 samples per type, so only large differences are detectable.
- The Small split resolves structures down to ~3 µm; globules below this are not separable at the derived 4.4 µm voxel size.
- `foreground_fraction` is the volume fraction of the segmented (mostly fat) phase, not porosity.
