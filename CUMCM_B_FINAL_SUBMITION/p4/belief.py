"""问题4的混合源后验。

一个后验同时保留全向解释和定向解释。定向源朝向用固定圆周网格表示；
位置支持域仍使用连续几何区域，因此清除失败可以精确扣除清除圆。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from P2 import Problem2Result
from utils import DetectionSector, Point, Region, minimum_enclosing_circle


@dataclass(frozen=True)
class MixedCandidateEvaluation:
    point: Point
    signal_probability: float
    clear_probability: float
    expected_radius: float
    worst_radius: float
    travel_distance: float


class MixedSourceBelief:
    """位置、接收半径、源类型和定向朝向的联合离散后验。"""

    def __init__(
        self,
        result: Problem2Result,
        *,
        bearing_error_deg: float,
        near_radius: float,
        clear_radius: float,
        orientation_step_deg: float = 10.0,
    ) -> None:
        self.result = result
        self.bearing_error_deg = bearing_error_deg
        self.near_radius = near_radius
        self.clear_radius = clear_radius
        self.orientations = np.arange(0.0, 360.0, orientation_step_deg)
        self.targets = np.array(
            [(state.target.x, state.target.y) for state in result.states], dtype=float
        )
        self.radii = np.array([state.receive_radius for state in result.states], dtype=float)
        self.target_indices = np.array([state.target_index for state in result.states], dtype=int)
        self.cell_radii = np.array([state.cell_radius for state in result.states], dtype=float)
        base = np.array([state.weight for state in result.states], dtype=float)
        self.omni_weights = base * 0.5
        self.directional_weights = np.repeat(
            (base * 0.5 / len(self.orientations))[:, None],
            len(self.orientations),
            axis=1,
        )
        self.position_region = result.first_feasible_region
        self.failed_clear_points: list[Point] = []
        self.version = 0
        self.inconsistent = False

    @classmethod
    def from_observations(
        cls,
        result: Problem2Result,
        observations: Iterable[object],
        *,
        bearing_error_deg: float,
        near_radius: float,
        clear_radius: float,
        orientation_step_deg: float = 10.0,
    ) -> "MixedSourceBelief":
        posterior = cls(
            result,
            bearing_error_deg=bearing_error_deg,
            near_radius=near_radius,
            clear_radius=clear_radius,
            orientation_step_deg=orientation_step_deg,
        )
        for observation in observations:
            posterior.update_measurement(observation)
        return posterior

    @property
    def total_weight(self) -> float:
        return float(self.omni_weights.sum() + self.directional_weights.sum())

    @property
    def omni_probability(self) -> float:
        total = self.total_weight
        return float(self.omni_weights.sum() / total) if total else 0.0

    def update_measurement(self, observation: object) -> bool:
        """用 direction、near 或 no_signal 同时更新两类源假设。"""
        kind = str(getattr(observation, "kind"))
        point = getattr(observation, "position")
        bearing = getattr(observation, "bearing_deg", None)
        dx = self.targets[:, 0] - point.x
        dy = self.targets[:, 1] - point.y
        distance = np.hypot(dx, dy)
        in_range = distance <= self.radii + 1e-9

        target_to_sensor = (np.degrees(np.arctan2(-dy, -dx)) + 360.0) % 360.0
        angle_delta = np.abs(
            (target_to_sensor[:, None] - self.orientations[None, :] + 180.0) % 360.0
            - 180.0
        )
        visible = angle_delta <= 90.0 + 1e-9

        omni = self.omni_weights.copy()
        directional = self.directional_weights.copy()
        if kind == "near":
            close = distance <= self.near_radius + 1e-9
            omni *= close
            directional *= close[:, None] & visible
        elif kind == "direction":
            if bearing is None:
                return False
            true_bearing = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
            bearing_delta = np.abs((true_bearing - float(bearing) + 180.0) % 360.0 - 180.0)
            cell_angle = np.degrees(
                np.arcsin(np.clip(self.cell_radii / np.maximum(distance, 1e-9), 0.0, 1.0))
            )
            compatible = in_range & (distance > self.near_radius) & (
                bearing_delta <= self.bearing_error_deg + 0.125 + cell_angle + 1e-9
            )
            omni *= compatible
            directional *= compatible[:, None] & visible
            sector = DetectionSector.from_measurement(
                point, float(bearing), self.bearing_error_deg
            ).to_region(self.result.config.receive_radius_max, self.near_radius)
            self.position_region = self.position_region.intersection(sector)
        elif kind == "no_signal":
            omni *= ~in_range
            directional *= (~in_range)[:, None] | ~visible
        else:
            return False

        total = float(omni.sum() + directional.sum())
        if total <= 0.0:
            self.inconsistent = True
            return False
        self.omni_weights = omni / total
        self.directional_weights = directional / total
        self.version += 1
        self.inconsistent = False
        return True

    def exclude_clear_disk(self, point: Point) -> None:
        """扣除一次失败清除圆，并保留精确的连续残余区域。"""
        disk = Region.disk(point, self.clear_radius)
        self.position_region = self.position_region.difference(disk)
        removed_targets = {
            cell.index for cell in self.result.target_cells if disk.covers(cell.geometry)
        }
        keep = ~np.isin(self.target_indices, list(removed_targets))
        self.omni_weights *= keep
        self.directional_weights *= keep[:, None]
        total = self.total_weight
        if total > 0.0:
            self.omni_weights /= total
            self.directional_weights /= total
        self.failed_clear_points.append(point)
        self.version += 1

    def evaluate_candidates(
        self,
        points: Iterable[Point],
        current_position: Point,
    ) -> list[MixedCandidateEvaluation]:
        return [self.evaluate_candidate(point, current_position) for point in points]

    def target_marginals(self) -> list[tuple[Point, float]]:
        """返回按目标位置合并接收半径与源类型后的权重。"""
        state_weights = self.omni_weights + self.directional_weights.sum(axis=1)
        combined: dict[int, tuple[Point, float]] = {}
        for index, weight in enumerate(state_weights):
            target_index = int(self.target_indices[index])
            point = Point(*self.targets[index])
            old_weight = combined.get(target_index, (point, 0.0))[1]
            if self.position_region.contains(point):
                combined[target_index] = point, old_weight + float(weight)
        return [item for item in combined.values() if item[1] > 1e-15]

    def evaluate_candidate(
        self,
        point: Point,
        current_position: Point,
    ) -> MixedCandidateEvaluation:
        """预测候选点的混合源接收率和观测后的定位尺度。"""
        dx = self.targets[:, 0] - point.x
        dy = self.targets[:, 1] - point.y
        distance = np.hypot(dx, dy)
        in_range = distance <= self.radii + 1e-9
        close = distance <= self.near_radius + 1e-9
        target_to_sensor = (np.degrees(np.arctan2(-dy, -dx)) + 360.0) % 360.0
        delta = np.abs(
            (target_to_sensor[:, None] - self.orientations[None, :] + 180.0) % 360.0
            - 180.0
        )
        visible = delta <= 90.0 + 1e-9

        omni_signal = self.omni_weights * in_range
        directional_signal = (self.directional_weights * (in_range[:, None] & visible)).sum(axis=1)
        signal_weight = omni_signal + directional_signal
        total = self.total_weight
        signal_probability = float(signal_weight.sum() / total) if total else 0.0
        clear_probability = float(signal_weight[close].sum() / total) if total else 0.0

        no_signal_weight = (
            self.omni_weights * ~in_range
            + (self.directional_weights * ((~in_range)[:, None] | ~visible)).sum(axis=1)
        )
        branch_radii: list[tuple[float, float]] = []
        if no_signal_weight.sum() > 0.0:
            branch_radii.append((float(no_signal_weight.sum()), self._support_radius(no_signal_weight)))
        bearings = ((np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0 / 5.0).astype(int)
        for bin_index in np.unique(bearings[signal_weight > 0.0]):
            weights = signal_weight * (bearings == bin_index) * ~close
            if weights.sum() > 0.0:
                branch_radii.append((float(weights.sum()), self._support_radius(weights)))
        if clear_probability > 0.0:
            branch_radii.append((clear_probability * total, self.near_radius))
        branch_total = sum(weight for weight, _ in branch_radii) or 1.0
        expected_radius = sum(weight * radius for weight, radius in branch_radii) / branch_total
        worst_radius = max((radius for _, radius in branch_radii), default=math.inf)
        return MixedCandidateEvaluation(
            point=point,
            signal_probability=signal_probability,
            clear_probability=clear_probability,
            expected_radius=expected_radius,
            worst_radius=worst_radius,
            travel_distance=current_position.distance_to(point),
        )

    def _support_radius(self, weights: np.ndarray) -> float:
        active = np.flatnonzero(weights > 1e-15)
        if not len(active):
            return 0.0
        by_target: dict[int, int] = {}
        for state_index in active:
            by_target.setdefault(int(self.target_indices[state_index]), int(state_index))
        indices = list(by_target.values())
        points = [Point(*self.targets[index]) for index in indices]
        circle = minimum_enclosing_circle(points)
        return circle.radius + max(float(self.cell_radii[index]) for index in indices)
