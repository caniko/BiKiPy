from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


@dataclass
class VideoMeta:
    fps: float
    total_frames: int
    resolution: tuple[int, int]
    meters_per_pixel: float
    path: str | None = None

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / self.fps if self.fps > 0 else 0.0

    def meters_to_pixels(self, value: float) -> float:
        return value / self.meters_per_pixel if self.meters_per_pixel > 0 else value

    def pixel_coords_from_metric(self, coords: np.ndarray) -> np.ndarray:
        return coords / self.meters_per_pixel if self.meters_per_pixel > 0 else coords


@dataclass
class CoordinateColumnSpec:
    x: str
    y: str


@dataclass
class HeuristicMeta:
    name: str
    result_column: str
    true_frames: int
    total_frames: int
    seconds: float

    @property
    def percentage(self) -> float:
        return (self.true_frames / self.total_frames * 100) if self.total_frames > 0 else 0.0


@dataclass
class PerimeterSpec:
    label: str
    shape: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float] | None:
        if "center_x" in self.params and "center_y" in self.params:
            return (self.params["center_x"], self.params["center_y"])
        if "vertices" in self.params:
            verts = np.array(self.params["vertices"])
            return tuple(verts.mean(axis=0))
        return None

    @property
    def vertices(self) -> np.ndarray | None:
        if "vertices" in self.params:
            return np.array(self.params["vertices"])
        return None


@dataclass
class InspectionSettings:
    minimum_seconds_tolerance: float
    maximum_seconds_distraction: float
    meters_per_pixel: float


@dataclass
class InspectionManifest:
    video: VideoMeta
    labels: list[str]
    label_colors: dict[str, str]
    coordinate_columns: dict[str, CoordinateColumnSpec]
    perimeters: list[PerimeterSpec]
    heuristics: list[HeuristicMeta]
    settings: InspectionSettings
    df: pl.DataFrame

    @classmethod
    def from_directory(cls, path: Path) -> InspectionManifest:
        """Load a manifest from a directory containing inspection JSON + Parquet files.

        Expects either:
        - A single `*_inspection.json` and `*_evaluated.parquet`, or
        - Specific filenames passed via `from_files`.
        """
        json_files = list(path.glob("*_inspection.json"))
        parquet_files = list(path.glob("*_evaluated.parquet"))

        if not json_files:
            msg = f"No *_inspection.json found in {path}"
            raise FileNotFoundError(msg)
        if not parquet_files:
            msg = f"No *_evaluated.parquet found in {path}"
            raise FileNotFoundError(msg)

        return cls.from_files(json_files[0], parquet_files[0])

    @classmethod
    def from_files(cls, json_path: Path, parquet_path: Path) -> InspectionManifest:
        """Load from explicit file paths."""
        with open(json_path) as f:
            raw = json.load(f)

        df = pl.read_parquet(parquet_path)

        video_raw = raw["video"]
        video = VideoMeta(
            fps=video_raw["fps"],
            total_frames=video_raw["total_frames"],
            resolution=tuple(video_raw["resolution"]),
            meters_per_pixel=video_raw["meters_per_pixel"],
            path=raw.get("video_path"),
        )

        coord_cols = {
            label: CoordinateColumnSpec(x=spec["x"], y=spec["y"])
            for label, spec in raw.get("coordinate_columns", {}).items()
        }

        perimeters = []
        for p in raw.get("perimeters", []):
            shape = p.pop("shape")
            label = p.pop("label")
            perimeters.append(PerimeterSpec(label=label, shape=shape, params=p))

        heuristics = [
            HeuristicMeta(
                name=h["name"],
                result_column=h["result_column"],
                true_frames=h["true_frames"],
                total_frames=h["total_frames"],
                seconds=h["seconds"],
            )
            for h in raw.get("heuristics", [])
        ]

        settings_raw = raw.get("settings", {})
        settings = InspectionSettings(
            minimum_seconds_tolerance=settings_raw.get("minimum_seconds_tolerance", 0.5),
            maximum_seconds_distraction=settings_raw.get("maximum_seconds_distraction", 1.0 / 3.0),
            meters_per_pixel=settings_raw.get("meters_per_pixel", 0.001),
        )

        return cls(
            video=video,
            labels=raw.get("labels", []),
            label_colors=raw.get("label_colors", {}),
            coordinate_columns=coord_cols,
            perimeters=perimeters,
            heuristics=heuristics,
            settings=settings,
            df=df,
        )

    def get_coordinates(self, label: str) -> np.ndarray:
        """Get (N, 2) coordinate array for a label."""
        spec = self.coordinate_columns[label]
        x = self.df[spec.x].to_numpy()
        y = self.df[spec.y].to_numpy()
        return np.column_stack([x, y])

    def get_boolean_index(self, heuristic_name: str) -> np.ndarray:
        """Get boolean array for a heuristic result column."""
        meta = next(h for h in self.heuristics if h.name == heuristic_name)
        return self.df[meta.result_column].to_numpy()

    def get_perimeter(self, label: str) -> PerimeterSpec:
        """Get perimeter spec by label."""
        return next(p for p in self.perimeters if p.label == label)
