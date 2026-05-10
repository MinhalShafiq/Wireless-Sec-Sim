"""Pydantic v2 schemas for scenarios, deployments, and runs.

A ``Scenario`` is a fully self-contained, YAML-loadable description of
one wireless experiment: which 3D environment, which transmitters and
receivers (including adversarial actors), the carrier band, and the
ray-tracing parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

TxRole = Literal["legitimate", "rogue", "jammer"]
RxRole = Literal["user", "eavesdropper"]


class Vec3(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float
    y: float
    z: float

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]


class TransmitterCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    position: Vec3
    power_dbm: float = 44.0
    role: TxRole = "legitimate"


class ReceiverCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    position: Vec3
    role: RxRole = "user"


class FrequencyBand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "n78"
    carrier_hz: float = 3.5e9
    bandwidth_hz: float = 100e6
    noise_figure_db: float = 7.0


class RadioMapCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cell_size_m: float = 5.0
    height_m: float = 1.5


class PropagationCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_depth: int = 3
    samples_per_tx: int = 1_000_000
    samples_per_src_paths: int = 500_000
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = True
    diffraction: bool = False
    seed: int = 42


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    scene: str
    band: FrequencyBand = Field(default_factory=FrequencyBand)
    transmitters: list[TransmitterCfg]
    receivers: list[ReceiverCfg] = []
    radio_map: RadioMapCfg | None = None
    propagation: PropagationCfg = Field(default_factory=PropagationCfg)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Scenario":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(mode="python"), f, sort_keys=False)
