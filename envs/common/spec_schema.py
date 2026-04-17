"""Pydantic v1 of spec.yaml. Loose intentionally — fields evolve as Gate A surfaces gaps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Meta(BaseModel):
    model_config = ConfigDict(extra="allow")
    year: int
    host: str
    problem_id: str
    domain: list[str]
    title: str
    official_points: float
    simulated_points: float
    scope_notes: str | None = None
    sources: dict[str, str] = Field(default_factory=dict)


class Apparatus(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    spec: str
    nominal_precision: dict[str, float] = Field(default_factory=dict)
    noise_sigma: dict[str, float] = Field(default_factory=dict)
    noise_model: str | None = None


class Parameter(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    description: str
    truth_value: float
    truth_sigma: float | None = None
    tolerance_full: float
    tolerance_half: float
    source: Literal["official", "derived"]


class PhysicsModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    family: str
    governing_equation: str
    relation_beta: str | None = None
    parameters: list[Parameter]


class Part(BaseModel):
    model_config = ConfigDict(extra="allow")
    part_id: str
    request: str
    answer: dict[str, Any] | None = None
    expected_data: dict[str, Any] | None = None
    points: float


class RubricItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    criterion: str
    points: float
    parameter: str | None = None
    min_samples: int | None = None


class Scoring(BaseModel):
    model_config = ConfigDict(extra="allow")
    total_points: float
    rubric: list[RubricItem]


class Spec(BaseModel):
    model_config = ConfigDict(extra="allow")
    meta: Meta
    apparatus: list[Apparatus]
    constants: dict[str, float]
    physics_model: PhysicsModel
    parts: list[Part]
    scoring: Scoring

    @classmethod
    def load(cls, path: str | Path) -> "Spec":
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)

    def param(self, name: str) -> Parameter:
        for p in self.physics_model.parameters:
            if p.name == name:
                return p
        raise KeyError(name)
