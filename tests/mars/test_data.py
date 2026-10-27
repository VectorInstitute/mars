"""Clip sampling and corpus role dispatch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mars.data.avid.config import AvidConfig
from mars.data.avid.dataset import AvidDataset
from mars.data.avid.manifest import (
    build as build_avid,
)
from mars.data.base import Sample
from mars.data.celebdf.config import CelebdfConfig
from mars.data.celebdf.dataset import CelebdfDataset
from mars.data.celebdf.manifest import (
    build as build_celebdf,
)
from mars.data.celebdf.manifest import (
    label_from_folder,
    official_to_repo_label,
    repo_label,
)
from mars.data.celebdf.manifest import (
    scan as scan_celebdf,
)
from mars.data.config import FAKE, REAL, DataConfig
from mars.data.dataset import ClipDataset, ConcatClipDataset, collate
from mars.data.df26.config import Df26Config
from mars.data.df26.dataset import Df26Dataset
from mars.data.df26.manifest import build as build_df26
from mars.data.dftimit.config import DftimitConfig
from mars.data.dftimit.dataset import DftimitDataset
from mars.data.dftimit.manifest import (
    build as build_dftimit,
)
from mars.data.dftimit.manifest import (
    parse_fake_name,
)
from mars.data.dftimit.manifest import (
    scan as scan_dftimit,
)
from mars.data.factory import build_split_datasets, dataset_class, load_records
from mars.data.ffpp.config import FfppConfig
from mars.data.ffpp.dataset import FfppDataset
from mars.data.ffpp.manifest import (
    build,
    resolve_splits_dir,
    split_from_dir,
)
from mars.data.genvideo.config import GenvideoConfig
from mars.data.genvideo.dataset import GenvideoDataset
from mars.data.genvideo.manifest import (
    build as build_genvideo,
)
from mars.data.genvideo.manifest import (
    scan as scan_genvideo,
)
from mars.data.gtav.config import GtavConfig
from mars.data.gtav.dataset import GtavDataset
from mars.data.gtav.manifest import (
    build as build_gtav,
)
from mars.data.gtav.manifest import (
    scan as scan_gtav,
)
from mars.data.hififace.config import HififaceConfig
from mars.data.hififace.dataset import HififaceOnlyfakeDataset, HififacePairedDataset
from mars.data.hififace.manifest import (
    build as build_hififace,
)
from mars.data.records import VideoRecord
from mars.data.sampling import clips_for_video, sampled_indices, segment
from mars.data.uadfv.config import UadfvConfig
from mars.data.uadfv.dataset import UadfvDataset
from mars.data.uadfv.manifest import (
    build as build_uadfv,
)


def test_sampling_indices_and_padding() -> None:
    assert sampled_indices(10, frame_stride=2) == [0, 2, 4, 6, 8]
    segs = segment(list(range(5)), n_frames=4, padding="last_frame")
    assert segs[0][0] == (0, 1, 2, 3)
    assert segs[1][0] == (4, 4, 4, 4)
    assert segs[1][1] == 1
    clips = clips_for_video("v", n_source_frames=8, n_frames=4, frame_stride=2)
    assert len(clips) == 1
    assert clips[0].frame_indices == (0, 2, 4, 6)


def test_factory_dispatches_ffpp() -> None:
    from mars.data.base import BaseDataset

    assert dataset_class("ffpp") is FfppDataset
    assert issubclass(FfppDataset, BaseDataset)


def test_factory_dispatches_celebdf() -> None:
    from mars.data.base import BaseDataset

    assert dataset_class("celebdf") is CelebdfDataset
    assert issubclass(CelebdfDataset, BaseDataset)


def test_factory_dispatches_dftimit() -> None:
    from mars.data.base import BaseDataset

    assert dataset_class("dftimit") is DftimitDataset
    assert issubclass(DftimitDataset, BaseDataset)


def test_factory_dispatches_avid() -> None:
    from mars.data.base import BaseDataset

    assert dataset_class("avid") is AvidDataset
    assert issubclass(AvidDataset, BaseDataset)


def test_factory_dispatches_uadfv() -> None:
    from mars.data.base import BaseDataset

    assert dataset_class("uadfv") is UadfvDataset
    assert issubclass(UadfvDataset, BaseDataset)


def test_factory_dispatches_genvideo() -> None:
    from mars.data.base import BaseDataset

    assert dataset_class("genvideo") is GenvideoDataset
    assert issubclass(GenvideoDataset, BaseDataset)


def test_factory_dispatches_df26() -> None:
    assert dataset_class("df26") is Df26Dataset


def test_factory_dispatches_hififace() -> None:
    from mars.data.base import BaseDataset

    assert dataset_class("hififace_paired") is HififacePairedDataset
    assert dataset_class("hififace_onlyfake") is HififaceOnlyfakeDataset
    assert issubclass(HififacePairedDataset, BaseDataset)
    assert issubclass(HififaceOnlyfakeDataset, BaseDataset)


def test_factory_dispatches_gtav() -> None:
    from mars.data.base import BaseDataset

    assert dataset_class("gtav") is GtavDataset
    assert issubclass(GtavDataset, BaseDataset)


def test_unknown_dataset_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mars.data.factory.load_corpus_yaml",
        lambda name: {"dataset": name, "roles": ["test"], "root": "/tmp"},
    )
    with pytest.raises(ValueError, match="unknown dataset"):
        load_records("not_a_corpus", DataConfig())
    with pytest.raises(ValueError, match="unknown dataset"):
        dataset_class("not_a_corpus", {"dataset": "not_a_corpus"})


def test_sample_extra_collate() -> None:
    def _item(method: str | None) -> Sample:
        extra = {"method": method} if method is not None else {}
        return Sample(
            frames=torch.zeros(2, 3, 4, 4),
            label=torch.tensor(0),
            n_valid=torch.tensor(2),
            video_index=torch.tensor(0),
            clip_index=torch.tensor(0),
            extra=extra,
        )

    batch = collate([_item("Deepfakes"), _item(None)])
    assert batch.frames.shape == (2, 2, 3, 4, 4)
    assert batch.label.shape == (2,)
    assert batch.extra["method"] == ["Deepfakes", None]


def test_test_only_corpus_cannot_train(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mars.data.factory.load_corpus_yaml",
        lambda name: {"dataset": name, "roles": ["test"], "root": "/tmp"},
    )
    with pytest.raises(ValueError, match="cannot use 'celebdf' as train"):
        build_split_datasets(DataConfig(train="celebdf", test=["celebdf"]), need=("train",))


def test_celebdf_yaml_cannot_train() -> None:
    with pytest.raises(ValueError, match="cannot use 'celebdf' as train"):
        build_split_datasets(DataConfig(train="celebdf", test=["celebdf"]), need=("train",))


def test_celebdf_folder_and_official_labels() -> None:
    assert label_from_folder("Celeb-synthesis") == FAKE
    assert label_from_folder("Celeb-real") == REAL
    assert label_from_folder("YouTube-real") == REAL
    assert official_to_repo_label(0) == FAKE
    assert official_to_repo_label(1) == REAL
    assert repo_label("Celeb-synthesis", 0) == FAKE
    assert repo_label("Celeb-real", 1) == REAL
    with pytest.raises(ValueError, match="maps to"):
        repo_label("Celeb-synthesis", 1, "Celeb-synthesis/x.mp4")
    with pytest.raises(ValueError, match="unknown Celeb-DF folder"):
        label_from_folder("Other")


def _write_celebdf_tree(root: Path, *, missing: str | None = None, swap: str | None = None) -> None:
    for folder in ("Celeb-real", "YouTube-real", "Celeb-synthesis"):
        (root / folder).mkdir()
    files = [
        ("1", "YouTube-real/00001.mp4"),
        ("1", "Celeb-real/id1_0000.mp4"),
        ("0", "Celeb-synthesis/id1_id0_0000.mp4"),
    ]
    lines: list[str] = []
    for lab, rel in files:
        if missing != rel:
            (root / rel).write_bytes(b"")
        listed = "1" if (swap == rel and lab == "0") else ("0" if swap == rel else lab)
        lines.append(f"{listed} {rel}")
    (root / "List_of_testing_videos.txt").write_text("\n".join(lines) + "\n")


def test_celebdf_scan_and_build(tmp_path: Path) -> None:
    _write_celebdf_tree(tmp_path)
    cfg = CelebdfConfig(root=str(tmp_path), manifest=str(tmp_path / "m.csv"))
    records = build_celebdf(cfg, lambda _path: 10)
    assert len(records) == 3
    by_id = {record.video_id: record for record in records}
    assert by_id["Celeb-synthesis/id1_id0_0000.mp4"].label == FAKE
    assert by_id["Celeb-real/id1_0000.mp4"].label == REAL
    assert by_id["YouTube-real/00001.mp4"].label == REAL
    assert by_id["YouTube-real/00001.mp4"].extra["method"] == "YouTube-real"
    assert {record.split for record in records} == {"test"}


def test_celebdf_missing_listed_file(tmp_path: Path) -> None:
    _write_celebdf_tree(tmp_path, missing="Celeb-real/id1_0000.mp4")
    cfg = CelebdfConfig(root=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="listed video missing"):
        scan_celebdf(cfg)


def test_celebdf_official_label_mismatch(tmp_path: Path) -> None:
    _write_celebdf_tree(tmp_path, swap="Celeb-synthesis/id1_id0_0000.mp4")
    cfg = CelebdfConfig(root=str(tmp_path))
    with pytest.raises(ValueError, match="official list label"):
        scan_celebdf(cfg)


def test_dftimit_parse_fake_name() -> None:
    assert parse_fake_name("si2203-video-fedw0.avi") == ("si2203", "fedw0")
    with pytest.raises(ValueError, match="unrecognized"):
        parse_fake_name("si2203.avi")


def _write_dftimit_tree(root: Path) -> None:
    real_sent = root / "VidTIMIT-Real" / "faks0" / "video" / "si2203"
    real_sent.mkdir(parents=True)
    (real_sent / "001").write_bytes(b"")
    (real_sent / "002").write_bytes(b"")
    head = root / "VidTIMIT-Real" / "faks0" / "video" / "head"
    head.mkdir()
    (head / "001").write_bytes(b"")
    hq = root / "DeepfakeTIMIT-Fake" / "higher_quality" / "faks0"
    lq = root / "DeepfakeTIMIT-Fake" / "lower_quality" / "faks0"
    hq.mkdir(parents=True)
    lq.mkdir(parents=True)
    (hq / "si2203-video-fedw0.avi").write_bytes(b"")
    (hq / "si2203.wav").write_bytes(b"")
    (lq / "si2203-video-fedw0.avi").write_bytes(b"")


def test_dftimit_scan_and_build(tmp_path: Path) -> None:
    _write_dftimit_tree(tmp_path)
    cfg = DftimitConfig(root=str(tmp_path), manifest=str(tmp_path / "m.csv"))
    records = build_dftimit(cfg, lambda _path: 10)
    assert len(records) == 3
    by_id = {record.video_id: record for record in records}
    assert by_id["real/faks0/si2203"].label == REAL
    assert by_id["real/faks0/si2203"].extra["method"] == "real"
    assert by_id["real/faks0/si2203"].extra["sentence"] == "si2203"
    assert by_id["higher_quality/faks0/si2203-video-fedw0"].label == FAKE
    assert by_id["higher_quality/faks0/si2203-video-fedw0"].extra["inserted"] == "fedw0"
    assert by_id["lower_quality/faks0/si2203-video-fedw0"].extra["method"] == "lower_quality"
    assert {record.split for record in records} == {"test"}
    assert all("head" not in record.video_id for record in records)


def test_dftimit_yaml_cannot_train() -> None:
    with pytest.raises(ValueError, match="cannot use 'dftimit' as train"):
        build_split_datasets(DataConfig(train="dftimit", test=["dftimit"]), need=("train",))


def test_dftimit_scan_skips_wav(tmp_path: Path) -> None:
    _write_dftimit_tree(tmp_path)
    cfg = DftimitConfig(root=str(tmp_path))
    entries = scan_dftimit(cfg)
    assert all(path.suffix != ".wav" for _, path, _, _ in entries)


def _write_avid_tree(root: Path) -> None:
    (root / "real").mkdir()
    (root / "fake").mkdir()
    (root / "real" / "01.mp4").write_bytes(b"")
    (root / "fake" / "01.mp4").write_bytes(b"")
    (root / "real" / "notes.txt").write_text("ignore")


def test_avid_scan_and_build(tmp_path: Path) -> None:
    _write_avid_tree(tmp_path)
    cfg = AvidConfig(root=str(tmp_path), manifest=str(tmp_path / "m.csv"))
    records = build_avid(cfg, lambda _path: 10)
    assert len(records) == 2
    by_id = {record.video_id: record for record in records}
    assert by_id["real/01.mp4"].label == REAL
    assert by_id["fake/01.mp4"].label == FAKE
    assert by_id["real/01.mp4"].extra["method"] == "real"
    assert by_id["fake/01.mp4"].extra["method"] == "fake"
    assert {record.split for record in records} == {"test"}


def test_avid_yaml_cannot_train() -> None:
    with pytest.raises(ValueError, match="cannot use 'avid' as train"):
        build_split_datasets(DataConfig(train="avid", test=["avid"]), need=("train",))


def _write_uadfv_tree(root: Path) -> None:
    (root / "real").mkdir()
    (root / "fake").mkdir()
    (root / "real" / "0000.mp4").write_bytes(b"")
    (root / "fake" / "0000_fake.mp4").write_bytes(b"")
    (root / "real" / "frames").mkdir()
    (root / "real" / "frames" / "0000.mp4").write_bytes(b"")
    (root / "fake" / "landmarks").mkdir()
    (root / "fake" / "landmarks" / "0000_fake.mp4").write_bytes(b"")


def test_uadfv_scan_and_build(tmp_path: Path) -> None:
    _write_uadfv_tree(tmp_path)
    cfg = UadfvConfig(root=str(tmp_path), manifest=str(tmp_path / "m.csv"))
    records = build_uadfv(cfg, lambda _path: 10)
    assert len(records) == 2
    by_id = {record.video_id: record for record in records}
    assert by_id["real/0000.mp4"].label == REAL
    assert by_id["fake/0000_fake.mp4"].label == FAKE
    assert by_id["real/0000.mp4"].extra["method"] == "real"
    assert by_id["fake/0000_fake.mp4"].extra["method"] == "fake"
    assert {record.split for record in records} == {"test"}
    assert all("frames" not in record.video_id and "landmarks" not in record.video_id for record in records)


def test_uadfv_yaml_cannot_train() -> None:
    with pytest.raises(ValueError, match="cannot use 'uadfv' as train"):
        build_split_datasets(DataConfig(train="uadfv", test=["uadfv"]), need=("train",))


def _write_genvideo_tree(root: Path) -> None:
    real = root / "Real"
    real.mkdir()
    for i in range(5):
        (real / f"r{i}.mp4").write_bytes(b"")
    crafter = root / "Fake" / "Crafter"
    crafter.mkdir(parents=True)
    for i in range(3):
        (crafter / f"c{i}.mp4").write_bytes(b"")
    sora = root / "Fake" / "Sora"
    sora.mkdir()
    (sora / "s0.mp4").write_bytes(b"")
    (sora / "s0.gif").write_bytes(b"")


def test_genvideo_scan_and_build(tmp_path: Path) -> None:
    _write_genvideo_tree(tmp_path)
    cfg = GenvideoConfig(
        root=str(tmp_path),
        generators=["Crafter", "Sora"],
        manifest=str(tmp_path / "m.csv"),
    )
    records = build_genvideo(cfg, lambda _path: 10)
    assert len(records) == 9
    by_id = {record.video_id: record for record in records}
    assert by_id["real/r0.mp4"].label == REAL
    assert by_id["real/r0.mp4"].extra["method"] == "real"
    assert by_id["Crafter/c0.mp4"].label == FAKE
    assert by_id["Crafter/c0.mp4"].extra["method"] == "Crafter"
    assert by_id["Sora/s0.mp4"].label == FAKE
    assert by_id["Sora/s0.mp4"].extra["method"] == "Sora"
    assert {record.split for record in records} == {"test"}
    assert all(not record.path.endswith(".gif") for record in records)


def test_genvideo_scan_skips_non_mp4(tmp_path: Path) -> None:
    _write_genvideo_tree(tmp_path)
    cfg = GenvideoConfig(root=str(tmp_path), generators=["Crafter", "Sora"])
    entries = scan_genvideo(cfg)
    assert all(path.suffix == ".mp4" for _, path, _, _ in entries)


def test_genvideo_cap_matches_reals_to_kept_fakes(tmp_path: Path) -> None:
    _write_genvideo_tree(tmp_path)
    cfg = GenvideoConfig(
        root=str(tmp_path),
        generators=["Crafter", "Sora"],
        max_videos_per_generator=2,
        manifest=str(tmp_path / "m.csv"),
    )
    records = build_genvideo(cfg, lambda _path: 10)
    methods = [record.extra["method"] for record in records]
    assert methods.count("Crafter") == 2
    assert methods.count("Sora") == 1
    assert methods.count("real") == 3
    assert len(records) == 6


def _write_df26_tree(root: Path) -> None:
    rows = [
        ("real/Studio/r0.mp4", "real", "real"),
        ("real/Studio/r1.mp4", "real", "real"),
        ("fake/Studio/Veo_3.1/f0.mp4", "fake", "Veo_3.1"),
        ("fake/Studio/Kling_3.0/f1.mp4", "fake", "Kling_3.0"),
    ]
    (root / "metadata").mkdir()
    with (root / "metadata" / "videos.csv").open("w") as handle:
        handle.write("video_path,label,generator\n")
        for path, label, generator in rows:
            handle.write(f"{path},{label},{generator}\n")


def test_df26_lists_videos_from_metadata(tmp_path: Path) -> None:
    _write_df26_tree(tmp_path)
    cfg = Df26Config(root=str(tmp_path), generators=["Veo_3.1", "Kling_3.0"])
    by_id = {record.video_id: record for record in build_df26(cfg, lambda _path: 10)}
    assert len(by_id) == 4
    assert by_id["real/Studio/r0.mp4"].label == REAL
    assert by_id["real/Studio/r0.mp4"].extra["method"] == "real"
    assert by_id["fake/Studio/Veo_3.1/f0.mp4"].label == FAKE
    assert by_id["fake/Studio/Veo_3.1/f0.mp4"].extra["method"] == "Veo_3.1"
    assert by_id["fake/Studio/Veo_3.1/f0.mp4"].path == str(tmp_path / "fake/Studio/Veo_3.1/f0.mp4")


def test_df26_generator_filter_keeps_every_real(tmp_path: Path) -> None:
    _write_df26_tree(tmp_path)
    cfg = Df26Config(root=str(tmp_path), generators=["Veo_3.1"])
    methods = [record.extra["method"] for record in build_df26(cfg, lambda _path: 10)]
    assert sorted(methods) == ["Veo_3.1", "real", "real"]


def test_df26_yaml_cannot_train() -> None:
    with pytest.raises(ValueError, match="cannot use 'df26' as train"):
        build_split_datasets(DataConfig(train="df26", test=["df26"]), need=("train",))


def test_genvideo_yaml_cannot_train() -> None:
    with pytest.raises(ValueError, match="cannot use 'genvideo' as train"):
        build_split_datasets(DataConfig(train="genvideo", test=["genvideo"]), need=("train",))


def _write_hififace_tree(root: Path, ffpp_root: Path) -> Path:
    for stem in ("000_003", "001_870", "002_006"):
        (root / f"{stem}.mp4").write_bytes(b"")
    videos = ffpp_root / "original_sequences" / "youtube" / "c23" / "videos"
    videos.mkdir(parents=True)
    for stem in ("000", "001"):
        (videos / f"{stem}.mp4").write_bytes(b"")
    split = ffpp_root / "test.json"
    split.write_text(json.dumps([["000", "001"]]))
    return split


def test_hififace_onlyfake_scan_and_build(tmp_path: Path) -> None:
    root = tmp_path / "hififace"
    root.mkdir()
    _write_hififace_tree(root, tmp_path / "ffpp")
    cfg = HififaceConfig(root=str(root), variant="onlyfake", manifest=str(tmp_path / "m.csv"))
    records = build_hififace(cfg, lambda _path: 10)
    assert len(records) == 3
    assert all(record.label == FAKE for record in records)
    assert {record.extra["method"] for record in records} == {"hififace"}
    assert {record.split for record in records} == {"test"}


def test_hififace_paired_scan_and_build(tmp_path: Path) -> None:
    root = tmp_path / "hififace"
    root.mkdir()
    ffpp_root = tmp_path / "ffpp"
    split = _write_hififace_tree(root, ffpp_root)
    cfg = HififaceConfig(
        root=str(root),
        variant="paired",
        ffpp_root=str(ffpp_root),
        split_file=str(split),
        manifest=str(tmp_path / "m.csv"),
    )
    records = build_hififace(cfg, lambda _path: 10)
    assert len(records) == 4
    by_id = {record.video_id: record for record in records}
    assert by_id["real/000.mp4"].label == REAL
    assert by_id["hififace/000_003.mp4"].label == FAKE
    assert by_id["real/001.mp4"].extra["method"] == "real"
    assert by_id["hififace/001_870.mp4"].extra["method"] == "hififace"
    assert "hififace/002_006.mp4" not in by_id
    assert {record.split for record in records} == {"test"}


def test_hififace_yaml_cannot_train() -> None:
    with pytest.raises(ValueError, match="cannot use 'hififace_paired' as train"):
        build_split_datasets(DataConfig(train="hififace_paired", test=["hififace_paired"]), need=("train",))
    with pytest.raises(ValueError, match="cannot use 'hififace_onlyfake' as train"):
        build_split_datasets(DataConfig(train="hififace_onlyfake", test=["hififace_onlyfake"]), need=("train",))


def _write_gtav_tree(root: Path) -> None:
    for name in ("seq_train", "seq_test", "seq_unused"):
        images = root / name / "images"
        images.mkdir(parents=True)
        (images / "000000.bmp").write_bytes(b"")
        (root / name / "visible").mkdir()
    split = root / "split"
    split.mkdir()
    (split / "split_train.txt").write_text("seq_train\n")
    (split / "split_val.txt").write_text("seq_test\n")


def test_gtav_scan_and_build(tmp_path: Path) -> None:
    _write_gtav_tree(tmp_path)
    cfg = GtavConfig(root=str(tmp_path), manifest=str(tmp_path / "m.csv"))
    records = build_gtav(cfg, lambda _path: 10)
    assert len(records) == 2
    by_id = {record.video_id: record for record in records}
    assert by_id["gtav/seq_train"].split == "train"
    assert by_id["gtav/seq_test"].split == "test"
    assert by_id["gtav/seq_train"].label == FAKE
    assert by_id["gtav/seq_test"].extra["method"] == "gtav"
    assert "gtav/seq_unused" not in by_id
    assert {record.path for record in records} == {
        str(tmp_path / "seq_train" / "images"),
        str(tmp_path / "seq_test" / "images"),
    }


def test_gtav_scan_skips_missing_images(tmp_path: Path) -> None:
    _write_gtav_tree(tmp_path)
    (tmp_path / "split" / "split_train.txt").write_text("seq_train\nseq_missing\n")
    cfg = GtavConfig(root=str(tmp_path), manifest=str(tmp_path / "m.csv"))
    names = [video_id for video_id, _, _, _, _ in scan_gtav(cfg)]
    assert names == ["gtav/seq_train", "gtav/seq_test"]
    records = build_gtav(cfg, lambda _path: 10)
    assert {record.video_id for record in records} == {"gtav/seq_train", "gtav/seq_test"}


def test_gtav_yaml_cannot_val() -> None:
    with pytest.raises(ValueError, match="cannot use 'gtav' as val"):
        build_split_datasets(DataConfig(train="gtav", val="gtav", test=["gtav"]), need=("val",))


def test_data_train_accepts_string_or_list() -> None:
    single = DataConfig.from_dict({"train": "ffpp", "test": "ffpp"})
    assert single.train == ["ffpp"]
    assert single.val == "ffpp"
    mixed = DataConfig.from_dict({"train": ["ffpp", "gtav"]})
    assert mixed.train == ["ffpp", "gtav"]
    assert mixed.val == "ffpp"


def test_concat_clip_dataset_prefixes_ids() -> None:
    cfg = DataConfig(n_frames=4, frame_stride=1, train="ffpp", test="ffpp")
    left = ClipDataset(
        [VideoRecord("a", "/tmp/a", REAL, 4, "train", extra={"method": "youtube"})],
        cfg,
        split="train",
    )
    left.name = "ffpp"
    right = ClipDataset(
        [VideoRecord("gtav/b", "/tmp/b", FAKE, 4, "train", extra={"method": "gtav"})],
        cfg,
        split="train",
    )
    right.name = "gtav"
    concat = ConcatClipDataset([left, right])
    assert concat.name == "ffpp+gtav"
    assert len(concat) == 2
    assert concat.labels == [REAL, FAKE]
    assert concat.video_ids == ["ffpp/a", "gtav/b"]


def _entries(*stems: str) -> list[tuple[str, Path, str, int]]:
    rows: list[tuple[str, Path, str, int]] = []
    for stem in stems:
        fake = "_" in stem
        rows.append((stem, Path(f"{stem}.mp4"), "Deepfakes" if fake else "youtube", FAKE if fake else REAL))
    return rows


def _write_splits(directory: Path, *, train: list, val: list, test: list) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train.json").write_text(json.dumps(train))
    (directory / "val.json").write_text(json.dumps(val))
    (directory / "test.json").write_text(json.dumps(test))


def test_official_split_assigns_original_and_fake(tmp_path: Path) -> None:
    _write_splits(
        tmp_path,
        train=[["071", "054"]],
        val=[["001", "002"]],
        test=[["003", "004"]],
    )
    assignment = split_from_dir(tmp_path, _entries("071", "054", "071_054", "001"))
    assert assignment["071"] == "train"
    assert assignment["054"] == "train"
    assert assignment["071_054"] == "train"
    assert assignment["001"] == "val"


def test_official_split_raises_on_cross_split(tmp_path: Path) -> None:
    _write_splits(
        tmp_path,
        train=[["071", "054"]],
        val=[["001", "002"]],
        test=[["003", "004"]],
    )
    with pytest.raises(ValueError, match="spans multiple official splits"):
        split_from_dir(tmp_path, _entries("071_001"))


def test_resolve_splits_dir_uses_package_dir_when_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundled = tmp_path / "splits"
    _write_splits(bundled, train=[["071", "054"]], val=[["001", "002"]], test=[["003", "004"]])
    monkeypatch.setattr("mars.data.ffpp.manifest.PACKAGE_SPLITS", bundled)
    assert resolve_splits_dir(FfppConfig(root="/tmp")) == bundled


def test_missing_splits_dir_falls_back_to_fractions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("mars.data.ffpp.manifest.PACKAGE_SPLITS", tmp_path / "missing")
    cfg = FfppConfig(root=str(tmp_path))
    assert resolve_splits_dir(cfg) is None
    entries = _entries("071", "054", "071_054", "080", "081", "080_081", "090", "091", "090_091")
    monkeypatch.setattr("mars.data.ffpp.manifest.scan", lambda _cfg: entries)
    records = build(cfg, lambda _path: 10)
    by_stem = {record.video_id.split("/", 1)[1]: record.split for record in records}
    assert by_stem["071"] == by_stem["054"] == by_stem["071_054"]
    assert set(by_stem.values()) <= {"train", "val", "test"}
    assert len(records) == len(entries)
