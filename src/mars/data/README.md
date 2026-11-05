# data

Corpora, manifests, clip sampling and decoding.

```
data/
├── base.py          # BaseDataset contract
├── dataset.py       # ClipDataset: manifest rows -> 64-frame clips + validity count
├── sampling.py      # frame stride, clip boundaries, padding
├── video.py         # decode frames from a video file or an image folder
├── transforms.py    # resize to image_size, scale to [-1, 1]
├── records.py       # VideoRecord
├── config.py        # DataConfig (train / val / test corpora, sampling)
├── factory.py       # corpus slug -> dataset class; enforces roles
├── loaders.py       # split loaders, samplers, class weights
├── ffpp/            # FaceForensics++          train / val / test
├── gtav/            # GTA-V (SAIL-VOS-3D)      train / test
├── celebdf/         # Celeb-DF v2              test
├── dftimit/         # DeepfakeTIMIT            test
├── uadfv/           # UADFV                    test
├── avid/            # AVID                     test
├── hififace/        # HifiFace paired, onlyfake test
├── genvideo/        # GenVideo-Val             test
├── vifbench/        # ViF-Bench (19 generators) test
└── df26/            # DF26                     test only (licence: evaluation only)
```

Each corpus package has `config.py` (fields from `configs/datasets/<slug>.yaml`),
`manifest.py` (scan the folder and write `artifacts/<slug>/manifest*.csv`), `dataset.py`
and `__main__.py` (build the manifest from the command line). A new corpus is one such
package plus its branch in `factory.py`.
