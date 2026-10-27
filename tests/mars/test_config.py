"""YAML merge, corpus files, and DataConfig."""

from __future__ import annotations

import pytest

from mars.config import load_corpus_yaml, load_yaml, parse_overrides
from mars.data.avid.config import AvidConfig
from mars.data.celebdf.config import CelebdfConfig
from mars.data.config import DataConfig
from mars.data.df26.config import Df26Config
from mars.data.dftimit.config import DftimitConfig
from mars.data.ffpp.config import FfppConfig
from mars.data.genvideo.config import GenvideoConfig
from mars.data.gtav.config import GtavConfig
from mars.data.hififace.config import HififaceConfig
from mars.data.uadfv.config import UadfvConfig
from mars.models.unite.config import load_config


def test_yaml_base_merge_and_set() -> None:
    raw = load_yaml(
        "configs/models/unite/config.yaml",
        parse_overrides(["train.epochs=3", "model.spatial_pool=attn"]),
    )
    assert raw["model"]["name"] == "unite"
    assert raw["train"]["epochs"] == 3
    assert raw["model"]["spatial_pool"] == "attn"
    assert raw["data"]["n_frames"] == 64
    assert raw["data"]["train"] == ["ffpp", "gtav"]
    expected_test = [
        "ffpp",
        "celebdf",
        "dftimit",
        "avid",
        "uadfv",
        "genvideo",
        "hififace_paired",
        "hififace_onlyfake",
        "gtav",
        "df26",
        "vifbench",
    ]
    assert raw["data"]["test"] == expected_test
    assert "root" not in raw["data"]
    assert "compression" not in raw["data"]
    cfg = load_config("configs/models/unite/config.yaml", parse_overrides(["train.epochs=2"]))
    assert cfg.model.name == "unite"
    assert cfg.train.epochs == 2
    assert cfg.data.train == ["ffpp", "gtav"]
    assert cfg.data.test == expected_test
    assert cfg.data.val == "ffpp"
    assert cfg.data.overrides == {}
    assert cfg.log.run_name  # from the model YAML; experiment names change
    cfg_named = load_config(
        "configs/models/unite/config.yaml",
        parse_overrides(["log.run_name=ci_check"]),
    )
    assert cfg_named.log.run_name == "ci_check"


def test_corpus_keys_stay_off_data_config() -> None:
    cfg = DataConfig.from_dict(
        {
            "train": "ffpp",
            "test": "ffpp",
            "manifest": "artifacts/ffpp/override.csv",
        }
    )
    assert cfg.train == ["ffpp"]
    assert cfg.test == ["ffpp"]
    assert cfg.overrides == {"manifest": "artifacts/ffpp/override.csv"}
    assert not hasattr(cfg, "root")

    with pytest.raises(ValueError, match="data.dataset was replaced"):
        DataConfig.from_dict({"dataset": "ffpp"})

    corpus = load_corpus_yaml("ffpp")
    assert corpus["roles"] == ["train", "val", "test"]
    assert corpus["root"].endswith("FF++/data")
    ffpp = FfppConfig.from_dict(
        {
            **corpus,
            "root": "/data/FF++/data",
            "originals": ["youtube"],
            "manipulations": ["Deepfakes"],
        }
    )
    assert ffpp.compression == "c23"
    assert ffpp.manipulations == ["Deepfakes"]
    assert "test" in ffpp.roles

    celebdf = load_corpus_yaml("celebdf")
    assert celebdf["roles"] == ["test"]
    assert celebdf["root"].endswith("Celeb-DF-V2")
    cfg_cdf = CelebdfConfig.from_dict(celebdf)
    assert cfg_cdf.roles == ["test"]
    assert cfg_cdf.list_file == "List_of_testing_videos.txt"

    dftimit = load_corpus_yaml("dftimit")
    assert dftimit["roles"] == ["test"]
    assert dftimit["root"].endswith("DeepfakeTIMIT")
    cfg_dft = DftimitConfig.from_dict(dftimit)
    assert cfg_dft.roles == ["test"]
    assert cfg_dft.qualities == ["higher_quality", "lower_quality"]

    avid = load_corpus_yaml("avid")
    assert avid["roles"] == ["test"]
    assert avid["root"].endswith("AVID")
    cfg_avid = AvidConfig.from_dict(avid)
    assert cfg_avid.roles == ["test"]
    assert cfg_avid.real_dir == "real"
    assert cfg_avid.fake_dir == "fake"

    uadfv = load_corpus_yaml("uadfv")
    assert uadfv["roles"] == ["test"]
    assert uadfv["root"].endswith("UADFV")
    cfg_uadfv = UadfvConfig.from_dict(uadfv)
    assert cfg_uadfv.roles == ["test"]
    assert cfg_uadfv.real_dir == "real"
    assert cfg_uadfv.fake_dir == "fake"

    genvideo = load_corpus_yaml("genvideo")
    assert genvideo["roles"] == ["test"]
    assert genvideo["root"].endswith("GenVideo-Val")
    cfg_gv = GenvideoConfig.from_dict(genvideo)
    assert cfg_gv.roles == ["test"]
    assert cfg_gv.real_dir == "Real"
    assert cfg_gv.fake_dir == "Fake"
    assert "Crafter" in cfg_gv.generators
    assert cfg_gv.max_videos_per_generator is None


def test_df26_yaml_is_evaluation_only() -> None:
    df26 = load_corpus_yaml("df26")
    cfg_df26 = Df26Config.from_dict(df26)
    assert cfg_df26.roles == ["test"]
    assert cfg_df26.root.endswith("DF26")
    assert len(cfg_df26.generators) == 10
    with pytest.raises(ValueError, match="evaluation-only"):
        Df26Config.from_dict({**df26, "roles": ["train", "test"]})


def test_hififace_corpus_yaml() -> None:
    paired = load_corpus_yaml("hififace_paired")
    assert paired["roles"] == ["test"]
    assert paired["root"].endswith("HifiFace")
    cfg_paired = HififaceConfig.from_dict(paired)
    assert cfg_paired.variant == "paired"
    assert cfg_paired.roles == ["test"]
    assert cfg_paired.compression == "c23"

    onlyfake = load_corpus_yaml("hififace_onlyfake")
    assert onlyfake["roles"] == ["test"]
    assert onlyfake["root"].endswith("HifiFace")
    cfg_only = HififaceConfig.from_dict(onlyfake)
    assert cfg_only.variant == "onlyfake"
    assert cfg_only.ffpp_root is None


def test_gtav_corpus_yaml() -> None:
    gtav = load_corpus_yaml("gtav")
    assert gtav["roles"] == ["train", "test"]
    assert gtav["root"].endswith("sailvos3d")
    cfg_gtav = GtavConfig.from_dict(gtav)
    assert cfg_gtav.roles == ["train", "test"]
    assert cfg_gtav.images_dir == "images"
    assert cfg_gtav.train_file == "split_train.txt"
    assert cfg_gtav.test_file == "split_val.txt"


def test_overrides_read_exponent_numbers_as_floats() -> None:
    parsed = parse_overrides(["train.lr=1e-5", "train.eps=1.0e-8", "train.epochs=10", "log.run_name=m_1e5"])
    assert parsed["train"] == {"lr": 1e-5, "eps": 1e-8, "epochs": 10}
    assert isinstance(parsed["train"]["lr"], float)
    assert parsed["log"]["run_name"] == "m_1e5"
