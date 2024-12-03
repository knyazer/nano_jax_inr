# ruff: noqa
import inspect
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from absl import logging
from functools import partial
from typing import List, Tuple, Union


class ConfigField:
    def parse(self):
        raise NotImplementedError

    def __repr__(self):
        return str(self)


class InrTypeEnum(ConfigField, Enum):
    STANDARD = auto()
    RELPOS = auto()


class DistFuncEnum(ConfigField, Enum):
    NEURBF = auto()
    SUMTO1_NONLINEAR = auto()
    MAX1_NONLINEAR = auto()
    MAX1 = auto()
    IDENTITY = auto()


class LatentInitEnum(ConfigField, Enum):
    UNIFORM = auto()


class HarmonicsMethodEnum(ConfigField, Enum):
    EXP = auto()


class HarmonicsFuncEnum(ConfigField, Enum):
    SIN = auto()
    COS = auto()


class LatentPosInitEnum(ConfigField, Enum):
    GRADIENT = auto()


class DecTypeEnum(ConfigField, Enum):
    RELU = auto()


class DatasetEnum(ConfigField, Enum):
    MNIST = auto()
    IMAGENETTE_FULLRES = auto()
    IMAGENETTE_CONSTANT = auto()
    CIFAR = auto()
    FASHIONMNIST = auto()


class TransformEnum(ConfigField, Enum):
    MNISTINCANVAS = auto()
    MIN1TO1 = auto()
    NORMALISE = auto()
    NORMALISE_MANUALLY = auto()


class SampleEnum(ConfigField, Enum):
    SUB = auto()
    ALL = auto()


class OptimEnum(ConfigField, Enum):
    ADAM = auto()


"""
config = Config(
    alias="",

    dataset = "fashionmnist_pngs_padded100centered",
    inr_type=InrTypeEnum.RELPOS,
    relpos_normalise=True,

    latent_dim=32,
    num_latents=40,
    num_neighbours=4,
    dist_function=DistFuncEnum.SUMTO1_NONLINEAR,
    latent_init_distr=(LatentInitEnum.UNIFORM, {"a": -0.0001, "b": 0.0001}),
    harmonics_method=(
        HarmonicsMethodEnum.EXP,
        {"start": 0.125, "end": 4096, "latent_dim": 32},
    ),
    harmonics_function=HarmonicsFuncEnum.COS,
    latent_position_init=(LatentPosInitEnum.GRADIENT, {"num_points": 40}),
    dec_share=True,
    dec_shared_at_step=500,
    dec_type=DecTypeEnum.RELU,
    dec_layers=["in", 1.0, "out"],
    out_dim=1,

    coord_subsampler=(SampleEnum.SUB, {"num_samples": 0.25, "pooled": True}),
    num_steps=1000,
    save_intermittently_at=[500],
    dec_optimiser=(OptimEnum.ADAM, {"lr": 0.005}),
    enc_optimiser=(OptimEnum.ADAM, {"lr": 0.005}),
    num_repeats=1,
)
"""


@dataclass
class Config:
    alias: str = "default_inr_alias"
    dataset: str = "default_dataset"

    inr_type: InrTypeEnum = InrTypeEnum.STANDARD
    relpos_normalise: bool | None = None
    latent_dim: int = 32
    num_latents: float = 40
    num_neighbours: int = 4
    dist_function: DistFuncEnum = DistFuncEnum.SUMTO1_NONLINEAR
    latent_init_distr: tuple[LatentInitEnum, dict] = field(
        default_factory=lambda: (LatentInitEnum.UNIFORM, {"a": -0.0001, "b": 0.0001})
    )
    harmonics_method: tuple[HarmonicsMethodEnum, dict] = field(
        default_factory=lambda: (
            HarmonicsMethodEnum.EXP,
            {"start": 0.125, "end": 4096, "latent_dim": 32},
        )
    )
    harmonics_function: HarmonicsFuncEnum = HarmonicsFuncEnum.COS
    latent_position_init: tuple[LatentPosInitEnum, dict] = field(
        default_factory=lambda: (LatentPosInitEnum.GRADIENT, {"num_points": 40})
    )

    dec_share: bool = True
    dec_shared_at_step: int = 500
    dec_type: DecTypeEnum = DecTypeEnum.RELU
    dec_layers: list = field(default_factory=lambda: ["in", 1.0, "out"])
    out_dim: int = 1

    coord_subsampler: tuple[SampleEnum, dict] = field(
        default_factory=lambda: (SampleEnum.SUB, {"num_samples": 0.25, "pooled": True})
    )
    num_steps: int = 1000
    save_intermittently_at: list[int] = field(default_factory=lambda: [500])
    dec_optimiser: tuple[OptimEnum, dict] = field(
        default_factory=lambda: (OptimEnum.ADAM, {"lr": 0.005})
    )
    enc_optimiser: tuple[OptimEnum, dict] = field(
        default_factory=lambda: (OptimEnum.ADAM, {"lr": 0.005})
    )
    num_repeats: int = 1


_CONFIG = Config()
_ind = 0
_sz = 0
_arange = []


def load_config_from_py(path: str):
    # loading the config file from the py file:
    # we just execute the file, and take out the 'config' variable
    # init namespace with current
    namespace = globals()
    with open(path, "r") as f:
        next(f)  # ignore first line cuz its import we don't have
        exec(f.read(), namespace)
    global _CONFIG
    _CONFIG = namespace["config"]
    global _sz, _arange
    _sz = (
        len(_CONFIG.latent_dim),
        len(_CONFIG.num_latents),
        len(_CONFIG.num_neighbours),
        len(_CONFIG.latent_position_init),
        len(_CONFIG.dec_layers),
    )
    _arange = np.array(np.unravel_index(np.arange(np.prod(_sz)), _sz)).T


def set_config(i):
    global _ind
    _ind = i

    logging.info(get_config().alias)


from copy import deepcopy


def get_config():
    config = deepcopy(_CONFIG)
    i, j, k, l, m = _arange[_ind]

    config.latent_dim = config.latent_dim[i]
    config.num_latents = config.num_latents[j]
    config.num_neighbours = config.num_neighbours[k]
    config.latent_position_init = config.latent_position_init[l]
    config.dec_layers = config.dec_layers[m]

    config.alias = (
        config.alias
        + "_"
        + str(config.latent_dim)
        + "_"
        + str(config.num_latents)
        + "_"
        + str(config.num_neighbours)
        + "_"
        + str(config.latent_position_init[1]["num_points"])
        + "_"
        + str(config.dec_layers[1])
    )
    return config
