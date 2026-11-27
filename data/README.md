# Data

This project uses the **MozzaVID Small** split. The dataset is not committed to the repository
(`.gitignore` excludes `Small/` and all TIFFs). Download it from the MozzaVID release and place
it so the pipeline can find it.

## Download

- Paper: [Pieta et al., *MozzaVID: Mozzarella Volumetric Image Dataset*, arXiv:2412.04880](https://arxiv.org/abs/2412.04880)
- Dataset card: `huggingface.co/datasets/dtudk/MozzaVID_*` (see the paper for the current link)

## Data root

The pipeline resolves the data root in this order (`io.resolve_data_root`):

1. the `--data-root` CLI flag,
2. the `MOZZAVID_DATA_ROOT` environment variable,
3. `./Small` (the default).

```pwsh
$env:MOZZAVID_DATA_ROOT = "D:\datasets\MozzaVID\Small"   # optional override
```

## Verified layout

```
Small/
  dataset_coarse.csv          # columns: file, label_cheese, label_sample, split
  dataset_fine.csv            # optional; label_sample is unique per sample
  cheese_<N>/                 # 25 cheese types, N = 1..25
    <sample_folder>/          # 149 sample folders total; NOT always contiguous
      <volume_idx>.tiff       # 1-4 volumes per sample; 192x192x192 float16 in [0,1]
```

- **591 TIFF volumes, 25 cheese types, 149 samples.** `load_metadata` verifies these counts and
  that the CSV `file` set matches the TIFFs on disk.
- **`sample_id` is the first two path components** (e.g. `cheese_4/3`), the independent statistical
  unit. It is derived from the folder path, not from the coarse CSV's `label_sample` (which is not
  unique: 99 distinct values for 149 folders).
- **Sample folders are not always contiguous.** For example `cheese_4` has folders `1,3,4,5,6,7`
  (the authors discarded sample 2). Never assume folder names `1..k`.
- **`volume_id` is the relative path** (e.g. `cheese_4/3/5.tiff`), never the bare volume index,
  because the index restarts per cheese type.
- The `split` column (ML train/val/test) is carried through but unused here.

## CSV columns

| Column | Meaning | Used |
|---|---|---|
| `file` | path relative to the data root, `/`-separated | yes (becomes `volume_id`) |
| `label_cheese` | integer cheese label; must equal `cheese_index - 1` | yes (validation) |
| `label_sample` | sample label; **not unique** in the coarse CSV | ignored (coarse); cross-checked (fine) |
| `split` | ML split | carried, unused |
