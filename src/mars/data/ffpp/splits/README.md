# FF++ split lists (optional)

This folder is where MARS looks for the official FaceForensics++ split lists. The `.json` files are
not shipped (they are ignored by git); download them if you need them.

| File | Used by |
|---|---|
| `train.json`, `val.json`, `test.json` | the FF++ manifest, for the official 720 / 140 / 140 identity split |
| `test.json` | the HifiFace paired corpus (see below) |

## Do I need them?

- **FF++ only:** no. Without `train.json` the manifest builder uses random identity-safe fractions
  (`split_fractions`, `split_seed` in `configs/datasets/ffpp.yaml`), which gives a different split from
  the official one, so FF++ numbers will not match the paper.
- **HifiFace paired:** yes. Its test videos are swaps whose target is in the official FF++ test
  split, so `hififace_paired` needs `test.json` and stops without it.

## Adding them

Download the three files from the FaceForensics++ repository into this folder:

```bash
cd src/mars/data/ffpp/splits
for s in train val test; do
  curl -O https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/$s.json
done
```

To keep them elsewhere, set `split_file:` to that directory in `configs/datasets/ffpp.yaml`
(and in `configs/datasets/hififace_paired.yaml`).

Delete `artifacts/ffpp/manifest.csv` (and the HifiFace manifest) before rebuilding: an existing
manifest is reused, not rebuilt.
