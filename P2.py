"""CUMCM 2026 B 题问题 2：第二检测点选择。

本模块把第一次测向后的连续可行域离散为带面积权重的六边形单元，并把
目标位置与未知接收半径组成联合状态。对每个第二检测候选点，程序完整处理
``near``、``direction`` 和 ``no_signal`` 三类观测，使用后验目标集合的最小
覆盖圆评价定位与一次清除能力。

默认优化方式是鲁棒优先：先比较最坏后验覆盖半径，再比较可清除情形比例和
移动成本。概率、均值和分位数只是离散模型下的辅助指标，不代表题目给定了
概率分布。Fisher 信息也只作为局部几何代理指标。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import cos, hypot, isfinite, log10, radians, sin, sqrt
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from utils import (
    Circle,
    DetectionSector,
    Point,
    Region,
    minimum_enclosing_circle,
)


ObservationKind = Literal["direction", "near", "no_signal"]
OptimizationMode = Literal["robust", "probabilistic"]


@dataclass(frozen=True)
class Problem2Config:
    """问题 2 的物理参数、离散精度和搜索范围。"""

    target_radius: float = 1800.0
    receive_radius_min: float = 1000.0
    receive_radius_max: float = 1500.0
    bearing_error_deg: float = 1.0
    near_radius: float = 5.0
    clear_radius: float = 20.0

    dog_speed: float = 5.0
    detection_time: float = 5.0
    clear_localize_time: float = 3.0
    clear_operation_time: float = 2.0

    # 20 m 邻点间距对应完整三角晶格约 11.55 m 的最大覆盖距离。
    target_grid_spacing: float = 20.0
    candidate_grid_spacing: float = 100.0
    refined_grid_spacing: float = 50
    refinement_radius: float = 160.0
    refinement_seed_count: int = 4
    candidate_margin: float = 300.0
    candidate_lateral_extent: float = 1000.0
    candidate_a_bounds: tuple[float, float] | None = None
    candidate_b_bounds: tuple[float, float] | None = None
    bilateral_search: bool = True

    receive_radius_sample_count: int = 5
    error_samples_deg: tuple[float, ...] = (-1.0, 0.0, 1.0)
    observation_bin_deg: float = 0.25
    circle_quad_segs: int = 64
    sector_arc_samples: int = 65

    optimization_mode: OptimizationMode = "robust"
    signal_fraction_power: float = 1
    signal_fraction_epsilon: float = 0.05
    require_guaranteed_signal: bool = False
    candidate_region_relative_tolerance: float = 0.05
    candidate_region_absolute_tolerance: float = 2.0
    probability_tolerance: float = 0.02
    random_seed: int = 2026

    def __post_init__(self) -> None:
        if not isfinite(self.signal_fraction_power) or self.signal_fraction_power < 0:
            raise ValueError("signal_fraction_power 必须为非负有限数。")
        if not isfinite(self.signal_fraction_epsilon) or not 0 < self.signal_fraction_epsilon <= 1:
            raise ValueError("signal_fraction_epsilon 必须位于 (0, 1]。")
        positive = {
            "target_radius": self.target_radius,
            "receive_radius_min": self.receive_radius_min,
            "receive_radius_max": self.receive_radius_max,
            "bearing_error_deg": self.bearing_error_deg,
            "near_radius": self.near_radius,
            "clear_radius": self.clear_radius,
            "dog_speed": self.dog_speed,
            "target_grid_spacing": self.target_grid_spacing,
            "candidate_grid_spacing": self.candidate_grid_spacing,
            "refined_grid_spacing": self.refined_grid_spacing,
            "refinement_radius": self.refinement_radius,
            "candidate_margin": self.candidate_margin,
            "candidate_lateral_extent": self.candidate_lateral_extent,
            "observation_bin_deg": self.observation_bin_deg,
        }
        invalid = [name for name, value in positive.items() if value <= 0 or not isfinite(value)]
        if invalid:
            raise ValueError(f"以下参数必须是正的有限数：{', '.join(invalid)}。")
        if self.receive_radius_min > self.receive_radius_max:
            raise ValueError("接收半径下限不能大于上限。")
        if self.near_radius > self.clear_radius:
            raise ValueError("near 半径不能大于清除半径。")
        if not 0.0 < self.bearing_error_deg < 90.0:
            raise ValueError("示向度误差必须位于 (0°, 90°) 内。")
        if self.receive_radius_sample_count < 2:
            raise ValueError("接收半径至少需要两个采样值。")
        if self.refinement_seed_count < 1:
            raise ValueError("局部加密种子数至少为 1。")
        if self.detection_time < 0 or self.clear_localize_time < 0 or self.clear_operation_time < 0:
            raise ValueError("各项操作时间不能为负。")
        if not self.error_samples_deg:
            raise ValueError("测角误差样本不能为空。")
        if any(abs(value) > self.bearing_error_deg + 1e-12 for value in self.error_samples_deg):
            raise ValueError("测角误差样本不能超出允许误差界。")
        if self.circle_quad_segs < 8 or self.sector_arc_samples < 3:
            raise ValueError("圆弧离散精度过低。")
        if self.optimization_mode not in ("robust", "probabilistic"):
            raise ValueError("optimization_mode 必须为 robust 或 probabilistic。")


@dataclass(frozen=True)
class TargetCell:
    """第一次可行域内一个被裁剪的六边形单元。"""

    index: int
    representative: Point
    geometry: Region
    area: float
    weight: float
    cell_radius: float


@dataclass(frozen=True)
class StateSample:
    """一个可能的目标位置—接收半径联合状态。"""

    target_index: int
    target: Point
    receive_radius: float
    weight: float
    cell_radius: float


@dataclass(frozen=True)
class Observation:
    """第二次检测可能返回的观测。"""

    kind: ObservationKind
    bearing_deg: float | None = None


@dataclass(frozen=True)
class EnclosingCircle:
    center: Point
    radius: float


@dataclass(frozen=True)
class PosteriorMetrics:
    observation: Observation
    event_weight: float
    state_count: int
    target_count: int
    clear_center: Point
    sample_radius: float
    conservative_radius: float
    can_clear: bool
    follow_up_distance: float
    follow_up_time: float


@dataclass
class CandidateMetrics:
    point: Point
    a: float
    b: float
    travel_distance: float
    guaranteed_signal: bool
    observation_count: int
    clear_fraction: float
    guaranteed_clear: bool
    expected_radius: float
    radius_q90: float
    worst_radius: float
    expected_total_time: float
    fisher_logdet: float
    signal_fraction: float = 0.0
    pareto_optimal: bool = False
    posterior_cases: list[PosteriorMetrics] = field(default_factory=list, repr=False)


@dataclass
class Problem2Result:
    config: Problem2Config
    first_position: Point
    first_bearing_deg: float
    first_feasible_region: Region
    target_cells: list[TargetCell]
    states: list[StateSample]
    safe_candidate_region: Region
    candidates: list[CandidateMetrics]
    best_candidate: CandidateMetrics
    pareto_candidates: list[CandidateMetrics]
    recommended_region: Region
    initial_worst_radius: float = float("inf")
    result_source: str = "direct_solve"


def _validate_inputs(first_position: Sequence[float], first_bearing_deg: float) -> Point:
    """校验第一次检测点和示向度，并返回统一的 ``Point``。"""
    if len(first_position) != 2:
        raise ValueError("第一次检测位置必须是 (x, y) 二元组。")
    x, y = float(first_position[0]), float(first_position[1])
    if not all(isfinite(value) for value in (x, y, first_bearing_deg)):
        raise ValueError("第一次检测位置和示向度必须是有限数值。")
    return Point(x, y)


def _wrap_angle_deg(angle: float) -> float:
    """把角度规范到 [-180°, 180°)。"""
    return (float(angle) + 180.0) % 360.0 - 180.0


def _angular_difference_deg(first: float | np.ndarray, second: float | np.ndarray):
    """返回 ``first - second`` 的最小有符号角差。"""
    return (np.asarray(first) - np.asarray(second) + 180.0) % 360.0 - 180.0


def _make_local_basis(bearing_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """建立第一次示向线的平行和左法向单位基。"""
    angle = radians(bearing_deg)
    return np.array([cos(angle), sin(angle)]), np.array([-sin(angle), cos(angle)])


def _local_to_global(
    first_position: Point,
    a: float,
    b: float,
    basis: tuple[np.ndarray, np.ndarray],
) -> Point:
    """把示向线局部坐标 (a, b) 转为全局坐标。"""
    vector = np.array([first_position.x, first_position.y]) + a * basis[0] + b * basis[1]
    return Point(float(vector[0]), float(vector[1]))


def _global_to_local(
    first_position: Point,
    point: Point,
    basis: tuple[np.ndarray, np.ndarray],
) -> tuple[float, float]:
    offset = np.array([point.x - first_position.x, point.y - first_position.y])
    return float(offset @ basis[0]), float(offset @ basis[1])


def _build_first_feasible_region(
    first_position: Point,
    first_bearing_deg: float,
    config: Problem2Config,
) -> Region:
    """构造目标圆域、距离约束与第一次方向带的连续交集。"""
    target_disk = Region.disk(
        Point(0.0, 0.0), config.target_radius, config.circle_quad_segs
    )
    sector = DetectionSector.from_measurement(
        first_position, first_bearing_deg, config.bearing_error_deg
    ).to_region(
        config.receive_radius_max,
        config.near_radius,
        config.sector_arc_samples,
    )
    region = target_disk.intersection(sector)
    if region.is_empty or region.area <= 0.0:
        raise ValueError("第一次观测与目标圆域没有非零面积的公共可行域。")
    return region


def _generate_triangular_lattice(
    bounds: tuple[float, float, float, float],
    spacing: float,
) -> list[tuple[float, float]]:
    """在包围盒及一圈余量内生成三角晶格中心。"""
    min_x, min_y, max_x, max_y = bounds
    row_height = sqrt(3.0) * spacing / 2.0
    min_y -= row_height
    max_y += row_height
    min_x -= spacing
    max_x += spacing
    rows = int(np.ceil((max_y - min_y) / row_height)) + 1
    columns = int(np.ceil((max_x - min_x) / spacing)) + 2
    points: list[tuple[float, float]] = []
    for row in range(rows):
        y = min_y + row * row_height
        shift = spacing / 2.0 if row % 2 else 0.0
        for column in range(columns):
            points.append((min_x + column * spacing + shift, y))
    return points


def _hexagon(center_x: float, center_y: float, spacing: float) -> Region:
    """返回三角晶格点的正六边形 Voronoi 单元。"""
    circumradius = spacing / sqrt(3.0)
    vertices = [
        Point(
            center_x + circumradius * cos(radians(30.0 + 60.0 * index)),
            center_y + circumradius * sin(radians(30.0 + 60.0 * index)),
        )
        for index in range(6)
    ]
    return Region.from_vertices(vertices)


def _geometry_coordinates(geometry: Region) -> list[tuple[float, float]]:
    """枚举几何对象边界上的坐标，用于计算代表点覆盖误差。"""
    return geometry.boundary_coordinates()


def _sample_target_cells(
    first_region: Region,
    config: Problem2Config,
) -> list[TargetCell]:
    """用裁剪六边形单元覆盖第一次连续可行域并按面积赋权。"""
    raw: list[tuple[Point, Region, float, float]] = []
    for center_x, center_y in _generate_triangular_lattice(
        first_region.bounds, config.target_grid_spacing
    ):
        clipped = _hexagon(center_x, center_y, config.target_grid_spacing).intersection(first_region)
        if clipped.is_empty or clipped.area <= 1e-9:
            continue
        representative = clipped.representative_point()
        cell_radius = max(
            hypot(x - representative.x, y - representative.y)
            for x, y in _geometry_coordinates(clipped)
        )
        raw.append((representative, clipped, float(clipped.area), float(cell_radius)))
    if not raw:
        raise RuntimeError("目标晶格没有覆盖第一次可行域，请检查网格参数。")
    total_area = sum(item[2] for item in raw)
    return [
        TargetCell(index, point, geometry, area, area / total_area, cell_radius)
        for index, (point, geometry, area, cell_radius) in enumerate(raw)
    ]


def _expand_receive_radius_states(
    target_cells: Sequence[TargetCell],
    first_position: Point,
    config: Problem2Config,
) -> list[StateSample]:
    """生成与第一次已收到信号相容的目标—接收半径联合状态。"""
    base_radii = np.linspace(
        config.receive_radius_min,
        config.receive_radius_max,
        config.receive_radius_sample_count,
    )
    states: list[StateSample] = []
    for cell in target_cells:
        first_distance = cell.representative.distance_to(first_position)
        lower = max(config.receive_radius_min, first_distance)
        radii = sorted({
            float(radius)
            for radius in (*base_radii, lower, config.receive_radius_max)
            if lower - 1e-9 <= radius <= config.receive_radius_max + 1e-9
        })
        if not radii:
            continue
        state_weight = cell.weight / len(radii)
        states.extend(
            StateSample(
                target_index=cell.index,
                target=cell.representative,
                receive_radius=radius,
                weight=state_weight,
                cell_radius=cell.cell_radius,
            )
            for radius in radii
        )
    total_weight = sum(state.weight for state in states)
    if total_weight <= 0.0:
        raise RuntimeError("没有与第一次检测相容的联合状态。")
    return [replace(state, weight=state.weight / total_weight) for state in states]


def _candidate_bounds(
    first_region: Region,
    first_position: Point,
    basis: tuple[np.ndarray, np.ndarray],
    config: Problem2Config,
) -> tuple[tuple[float, float], tuple[float, float]]:
    hull_coordinates = first_region.convex_hull.boundary_coordinates()
    local = [
        _global_to_local(first_position, Point(float(x), float(y)), basis)
        for x, y in hull_coordinates
    ]
    a_values = [value[0] for value in local]
    b_values = [value[1] for value in local]
    a_bounds = config.candidate_a_bounds or (
        min(0.0, min(a_values)) - config.candidate_margin,
        max(0.0, max(a_values)) + config.candidate_margin,
    )
    lateral_bound = max(
        max(abs(value) for value in b_values) + config.candidate_margin,
        config.candidate_lateral_extent,
    )
    b_bounds = config.candidate_b_bounds or (
        -lateral_bound if config.bilateral_search else 0.0,
        lateral_bound,
    )
    return a_bounds, b_bounds


def _inclusive_range(lower: float, upper: float, step: float) -> np.ndarray:
    count = max(1, int(np.floor((upper - lower) / step)))
    values = lower + np.arange(count + 1) * step
    if values[-1] < upper - 1e-9:
        values = np.append(values, upper)
    return values


def _generate_candidate_grid(
    first_region: Region,
    first_position: Point,
    first_bearing_deg: float,
    config: Problem2Config,
    spacing: float | None = None,
    windows: Sequence[tuple[float, float, float]] | None = None,
) -> list[tuple[Point, float, float]]:
    """在局部坐标中生成全域粗网格或若干局部细网格。"""
    basis = _make_local_basis(first_bearing_deg)
    step = spacing or config.candidate_grid_spacing
    if windows is None:
        a_bounds, b_bounds = _candidate_bounds(first_region, first_position, basis, config)
        windows = [
            (
                (a_bounds[0] + a_bounds[1]) / 2.0,
                (b_bounds[0] + b_bounds[1]) / 2.0,
                max(a_bounds[1] - a_bounds[0], b_bounds[1] - b_bounds[0]) / 2.0,
            )
        ]
        explicit_bounds = (a_bounds, b_bounds)
    else:
        explicit_bounds = None

    unique: dict[tuple[int, int], tuple[Point, float, float]] = {}
    for center_a, center_b, radius in windows:
        if explicit_bounds is None:
            a_bounds = (center_a - radius, center_a + radius)
            b_bounds = (center_b - radius, center_b + radius)
        else:
            a_bounds, b_bounds = explicit_bounds
        for a in _inclusive_range(*a_bounds, step):
            for b in _inclusive_range(*b_bounds, step):
                if not config.bilateral_search and config.candidate_b_bounds is None and b < -1e-9:
                    continue
                point = _local_to_global(first_position, float(a), float(b), basis)
                if point.distance_to(first_position) <= 1e-9:
                    continue  # 同一位置重复检测的误差不会改变。
                key = (round(float(a) / step * 1_000_000), round(float(b) / step * 1_000_000))
                unique[key] = (point, float(a), float(b))
    return list(unique.values())


def _build_safe_candidate_region(
    first_region: Region,
    config: Problem2Config,
) -> Region:
    """近似计算保证距每个可行目标不超过最小接收半径的区域。"""
    coordinates = first_region.convex_hull.boundary_coordinates()[:-1]
    safe: Region | None = None
    for x, y in coordinates:
        disk = Region.disk(
            Point(x, y), config.receive_radius_min,
            max(16, config.circle_quad_segs // 2),
        )
        safe = disk if safe is None else safe.intersection(disk)
        if safe.is_empty:
            return Region.empty()
    return safe if safe is not None else Region.empty()


def _simulate_observation(
    candidate: Point,
    state: StateSample,
    error_deg: float,
    config: Problem2Config,
) -> Observation:
    """由真实联合状态生成一次第二检测结果。"""
    distance = candidate.distance_to(state.target)
    if distance <= config.near_radius:
        return Observation("near")
    if distance > state.receive_radius:
        return Observation("no_signal")
    bearing = candidate.bearing_to(state.target)
    return Observation("direction", (bearing + error_deg) % 360.0)


def _observation_key(observation: Observation, config: Problem2Config) -> tuple[str, float | None]:
    if observation.kind != "direction":
        return observation.kind, None
    assert observation.bearing_deg is not None
    rounded = round(observation.bearing_deg / config.observation_bin_deg) * config.observation_bin_deg
    return "direction", rounded % 360.0


def _state_arrays(states: Sequence[StateSample]) -> dict[str, np.ndarray]:
    return {
        "x": np.array([state.target.x for state in states]),
        "y": np.array([state.target.y for state in states]),
        "radius": np.array([state.receive_radius for state in states]),
        "weight": np.array([state.weight for state in states]),
        "target_index": np.array([state.target_index for state in states], dtype=int),
        "cell_radius": np.array([state.cell_radius for state in states]),
    }


def _is_observation_compatible(
    arrays: dict[str, np.ndarray],
    candidate: Point,
    observation: Observation,
    config: Problem2Config,
) -> np.ndarray:
    """向量化判断所有状态是否可能产生给定观测。"""
    dx = arrays["x"] - candidate.x
    dy = arrays["y"] - candidate.y
    distances = np.hypot(dx, dy)
    if observation.kind == "near":
        return distances <= config.near_radius + 1e-9
    if observation.kind == "no_signal":
        return distances > arrays["radius"] + 1e-9
    assert observation.bearing_deg is not None
    bearings = np.degrees(np.arctan2(dy, dx)) % 360.0
    angular_tolerance = config.bearing_error_deg + config.observation_bin_deg / 2.0
    return (
        (distances > config.near_radius + 1e-9)
        & (distances <= arrays["radius"] + 1e-9)
        & (np.abs(_angular_difference_deg(bearings, observation.bearing_deg)) <= angular_tolerance + 1e-12)
    )


def _update_posterior_states(
    states: Sequence[StateSample],
    arrays: dict[str, np.ndarray],
    candidate: Point,
    observation: Observation,
    config: Problem2Config,
) -> tuple[list[StateSample], np.ndarray]:
    """保留与第二次观测相容的联合状态。"""
    mask = _is_observation_compatible(arrays, candidate, observation, config)
    indices = np.flatnonzero(mask)
    posterior = [states[int(index)] for index in indices]
    return posterior, indices


def _project_target_points(states: Sequence[StateSample]) -> tuple[list[Point], list[float]]:
    """把联合状态投影到不重复的目标位置及其单元误差。"""
    targets: dict[int, tuple[Point, float]] = {}
    for state in states:
        targets[state.target_index] = (state.target, state.cell_radius)
    ordered = [targets[index] for index in sorted(targets)]
    return [item[0] for item in ordered], [item[1] for item in ordered]


def _minimum_enclosing_circle(
    points: Sequence[Point],
    random_seed: int = 2026,
) -> EnclosingCircle:
    """兼容 P3 现有导入；实现已统一移动到 utils。"""
    circle: Circle = minimum_enclosing_circle(points, random_seed)
    return EnclosingCircle(circle.center, circle.radius)


def _bearing_gradient(sensor: Point, target: Point) -> np.ndarray:
    dx, dy = target.x - sensor.x, target.y - sensor.y
    radius_squared = dx * dx + dy * dy
    if radius_squared <= 1e-18:
        raise ValueError("检测点和目标重合时方位角梯度无定义。")
    return np.array([-dy / radius_squared, dx / radius_squared])


def _fisher_information_score(
    first_position: Point,
    second_position: Point,
    target: Point,
    sigma_theta: float,
) -> tuple[float, float, float]:
    """返回 FIM 的 log10 行列式、A 指标和两视线锐交会角。"""
    try:
        first_gradient = _bearing_gradient(first_position, target)
        second_gradient = _bearing_gradient(second_position, target)
    except ValueError:
        return float("-inf"), float("inf"), 0.0
    information = (
        np.outer(first_gradient, first_gradient)
        + np.outer(second_gradient, second_gradient)
    ) / (sigma_theta * sigma_theta)
    determinant = float(np.linalg.det(information))
    if determinant <= 1e-30:
        logdet, a_score = float("-inf"), float("inf")
    else:
        logdet = log10(determinant)
        a_score = float(np.trace(np.linalg.inv(information)))
    first_bearing = first_position.bearing_to(target)
    second_bearing = second_position.bearing_to(target)
    angle = abs(float(_angular_difference_deg(first_bearing, second_bearing)))
    return logdet, a_score, min(angle, 180.0 - angle)


def _evaluate_posterior(
    posterior_states: Sequence[StateSample],
    candidate: Point,
    observation: Observation,
    event_weight: float,
    config: Problem2Config,
) -> PosteriorMetrics:
    points, cell_radii = _project_target_points(posterior_states)
    if not points:
        raise RuntimeError("模拟得到的观测没有任何相容后验状态。")
    if observation.kind == "near":
        circle = EnclosingCircle(candidate, config.near_radius)
        conservative_radius = config.near_radius
        follow_up_distance = 0.0
    else:
        circle = _minimum_enclosing_circle(points, config.random_seed)
        conservative_radius = circle.radius + max(cell_radii, default=0.0)
        follow_up_distance = candidate.distance_to(circle.center)
    can_clear = conservative_radius <= config.clear_radius + 1e-9
    follow_up_time = follow_up_distance / config.dog_speed
    if can_clear:
        follow_up_time += config.clear_localize_time + config.clear_operation_time
    return PosteriorMetrics(
        observation=observation,
        event_weight=event_weight,
        state_count=len(posterior_states),
        target_count=len(points),
        clear_center=circle.center,
        sample_radius=circle.radius,
        conservative_radius=conservative_radius,
        can_clear=can_clear,
        follow_up_distance=follow_up_distance,
        follow_up_time=follow_up_time,
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    ordered_values, ordered_weights = values[order], weights[order]
    cumulative = np.cumsum(ordered_weights) / np.sum(ordered_weights)
    return float(ordered_values[min(np.searchsorted(cumulative, quantile), len(values) - 1)])


def _evaluate_candidate(
    point: Point,
    a: float,
    b: float,
    first_position: Point,
    target_cells: Sequence[TargetCell],
    states: Sequence[StateSample],
    arrays: dict[str, np.ndarray],
    safe_region: Region,
    config: Problem2Config,
) -> CandidateMetrics:
    """枚举一个候选点的可观测事件并汇总后验定位指标。"""
    events: dict[tuple[str, float | None], float] = {}
    error_weight = 1.0 / len(config.error_samples_deg)
    for state in states:
        for error in config.error_samples_deg:
            observation = _simulate_observation(point, state, error, config)
            key = _observation_key(observation, config)
            events[key] = events.get(key, 0.0) + state.weight * error_weight

    posterior_cases: list[PosteriorMetrics] = []
    for (kind, bearing), event_weight in events.items():
        observation = Observation(kind, bearing)  # type: ignore[arg-type]
        posterior_states, _ = _update_posterior_states(
            states, arrays, point, observation, config
        )
        if posterior_states:
            posterior_cases.append(
                _evaluate_posterior(
                    posterior_states, point, observation, event_weight, config
                )
            )
    if not posterior_cases:
        raise RuntimeError("候选点没有产生任何有效观测事件。")

    total_event_weight = sum(events.values())
    signal_fraction = sum(
        event_weight
        for (kind, _), event_weight in events.items()
        if kind != "no_signal"
    ) / total_event_weight
    weights = np.array([case.event_weight for case in posterior_cases])
    weights /= weights.sum()
    radii = np.array([case.conservative_radius for case in posterior_cases])
    clear = np.array([case.can_clear for case in posterior_cases], dtype=float)
    travel_distance = point.distance_to(first_position)
    base_time = travel_distance / config.dog_speed + config.detection_time
    expected_total_time = base_time + float(
        np.dot(weights, np.array([case.follow_up_time for case in posterior_cases]))
    )

    sigma_theta = radians(config.bearing_error_deg / sqrt(3.0))
    fisher_values: list[float] = []
    fisher_weights: list[float] = []
    for cell in target_cells:
        value, _, _ = _fisher_information_score(
            first_position, point, cell.representative, sigma_theta
        )
        if isfinite(value):
            fisher_values.append(value)
            fisher_weights.append(cell.weight)
    fisher_logdet = (
        float(np.average(fisher_values, weights=fisher_weights))
        if fisher_values else float("-inf")
    )
    guaranteed_signal = not safe_region.is_empty and safe_region.covers(point)
    return CandidateMetrics(
        point=point,
        a=a,
        b=b,
        travel_distance=travel_distance,
        guaranteed_signal=guaranteed_signal,
        observation_count=len(posterior_cases),
        clear_fraction=float(np.dot(weights, clear)),
        guaranteed_clear=bool(np.all(clear)),
        expected_radius=float(np.dot(weights, radii)),
        radius_q90=_weighted_quantile(radii, weights, 0.9),
        worst_radius=float(np.max(radii)),
        expected_total_time=expected_total_time,
        fisher_logdet=fisher_logdet,
        signal_fraction=float(np.clip(signal_fraction, 0.0, 1.0)),
        posterior_cases=posterior_cases,
    )


def adjusted_worst_radius(candidate: CandidateMetrics, config: Problem2Config) -> float:
    """接收率偏置评分；不改变原始半径、状态权重或连续接收保证。"""
    if config.signal_fraction_power == 0:
        return candidate.worst_radius
    denominator = max(candidate.signal_fraction, config.signal_fraction_epsilon) ** config.signal_fraction_power
    return candidate.worst_radius / denominator if denominator else float("inf")


def _candidate_sort_key(candidate: CandidateMetrics, config: Problem2Config) -> tuple[float, ...]:
    prefix = (float(not candidate.guaranteed_signal),) if config.require_guaranteed_signal else ()
    if config.optimization_mode == "robust":
        return prefix + (
            adjusted_worst_radius(candidate, config),
            -candidate.clear_fraction,
            candidate.radius_q90,
            candidate.expected_radius,
            candidate.expected_total_time,
        )
    return prefix + (
        -candidate.clear_fraction,
        candidate.expected_radius,
        candidate.radius_q90,
        candidate.expected_total_time,
        candidate.worst_radius,
    )


def _evaluate_all_candidates(
    raw_candidates: Sequence[tuple[Point, float, float]],
    first_position: Point,
    target_cells: Sequence[TargetCell],
    states: Sequence[StateSample],
    safe_region: Region,
    config: Problem2Config,
) -> list[CandidateMetrics]:
    arrays = _state_arrays(states)
    results = [
        _evaluate_candidate(
            point, a, b, first_position, target_cells, states, arrays, safe_region, config
        )
        for point, a, b in raw_candidates
    ]
    return sorted(results, key=lambda item: _candidate_sort_key(item, config))


def _refine_candidate_grid(
    coarse_results: Sequence[CandidateMetrics],
    first_region: Region,
    first_position: Point,
    first_bearing_deg: float,
    config: Problem2Config,
) -> list[tuple[Point, float, float]]:
    seeds = coarse_results[: config.refinement_seed_count]
    windows = [(seed.a, seed.b, config.refinement_radius) for seed in seeds]
    return _generate_candidate_grid(
        first_region,
        first_position,
        first_bearing_deg,
        config,
        spacing=config.refined_grid_spacing,
        windows=windows,
    )


def _dominates(first: CandidateMetrics, second: CandidateMetrics) -> bool:
    first_values = (
        first.worst_radius,
        -first.clear_fraction,
        first.expected_radius,
        first.expected_total_time,
    )
    second_values = (
        second.worst_radius,
        -second.clear_fraction,
        second.expected_radius,
        second.expected_total_time,
    )
    return all(a <= b + 1e-10 for a, b in zip(first_values, second_values)) and any(
        a < b - 1e-10 for a, b in zip(first_values, second_values)
    )


def _find_pareto_candidates(candidates: Sequence[CandidateMetrics]) -> list[CandidateMetrics]:
    """标记覆盖半径、清除比例和时间四指标下的非支配点。"""
    pareto: list[CandidateMetrics] = []
    for candidate in candidates:
        if not any(_dominates(other, candidate) for other in candidates if other is not candidate):
            candidate.pareto_optimal = True
            pareto.append(candidate)
    return pareto


def _extract_candidate_region(
    candidates: Sequence[CandidateMetrics],
    config: Problem2Config,
) -> Region:
    """把相对最优的细网格点合并成稳定候选区域。"""
    best = candidates[0]
    def score(candidate: CandidateMetrics) -> float:
        return (adjusted_worst_radius(candidate, config)
                if config.optimization_mode == "robust" else candidate.worst_radius)
    radius_limit = (
        score(best) * (1.0 + config.candidate_region_relative_tolerance)
        + config.candidate_region_absolute_tolerance
    )
    selected = [
        candidate
        for candidate in candidates
        if (not config.require_guaranteed_signal or candidate.guaranteed_signal)
        and score(candidate) <= radius_limit
        and candidate.clear_fraction >= best.clear_fraction - config.probability_tolerance
    ]
    if not selected:
        selected = [best]
    cell_radius = config.refined_grid_spacing / sqrt(2.0)
    return Region.union_all(
        Region.disk(candidate.point, cell_radius, quad_segs=4)
        for candidate in selected
    )


def solve_problem_2(
    first_position: Sequence[float],
    first_bearing_deg: float,
    config: Problem2Config | None = None,
    *,
    _first_region: Region | None = None,
) -> Problem2Result:
    """求第二检测点、Pareto 点集及近优候选区域。"""
    active_config = config or Problem2Config()
    first = _validate_inputs(first_position, first_bearing_deg)
    region = (_first_region if _first_region is not None else
              _build_first_feasible_region(first, first_bearing_deg, active_config))
    cells = _sample_target_cells(region, active_config)
    states = _expand_receive_radius_states(cells, first, active_config)
    safe_region = _build_safe_candidate_region(region, active_config)
    if active_config.require_guaranteed_signal and safe_region.is_empty:
        raise ValueError("保证接收区域为空，无法在其内部选择第二检测点。")

    coarse_grid = _generate_candidate_grid(
        region, first, first_bearing_deg, active_config
    )
    coarse_results = _evaluate_all_candidates(
        coarse_grid, first, cells, states, safe_region, active_config
    )
    fine_grid = _refine_candidate_grid(
        coarse_results, region, first, first_bearing_deg, active_config
    )
    fine_results = _evaluate_all_candidates(
        fine_grid, first, cells, states, safe_region, active_config
    )

    combined: dict[tuple[float, float], CandidateMetrics] = {
        (round(item.point.x, 8), round(item.point.y, 8)): item
        for item in (*coarse_results, *fine_results)
    }
    candidates = sorted(
        combined.values(),
        key=lambda item: _candidate_sort_key(item, active_config),
    )
    if active_config.require_guaranteed_signal:
        interior_candidates = [
            candidate for candidate in candidates
            if candidate.guaranteed_signal
            and safe_region.covers(Region.disk(candidate.point, 1e-6))
        ]
        if not interior_candidates:
            raise ValueError("当前网格未采到保证接收区域内部点，请减小网格间距或扩大搜索边界。")
        best_interior = interior_candidates[0]
        candidates = [best_interior] + [candidate for candidate in candidates if candidate is not best_interior]
    pareto = _find_pareto_candidates(candidates)
    recommended_region = _extract_candidate_region(candidates, active_config)
    prior_circle = _minimum_enclosing_circle(
        [cell.representative for cell in cells], active_config.random_seed
    )
    initial_worst_radius = prior_circle.radius + max(
        (cell.cell_radius for cell in cells), default=0.0
    )
    return Problem2Result(
        config=active_config,
        first_position=first,
        first_bearing_deg=float(first_bearing_deg) % 360.0,
        first_feasible_region=region,
        target_cells=cells,
        states=states,
        safe_candidate_region=safe_region,
        candidates=candidates,
        best_candidate=candidates[0],
        pareto_candidates=pareto,
        recommended_region=recommended_region,
        initial_worst_radius=initial_worst_radius,
    )


def evaluate_problem_2_candidate(
    result: Problem2Result,
    point: Point | Sequence[float],
) -> CandidateMetrics:
    """复用已求得的 P2 状态，对任意全局坐标执行完整后验评价。"""
    candidate = point if isinstance(point, Point) else Point(float(point[0]), float(point[1]))
    a, b = _global_to_local(
        result.first_position,
        candidate,
        _make_local_basis(result.first_bearing_deg),
    )
    return _evaluate_candidate(
        candidate,
        a,
        b,
        result.first_position,
        result.target_cells,
        result.states,
        _state_arrays(result.states),
        result.safe_candidate_region,
        result.config,
    )


def run_convergence_check(
    first_position: Sequence[float],
    first_bearing_deg: float,
    configs: Sequence[Problem2Config],
) -> list[dict[str, float]]:
    """用多套离散精度重复求解，返回最优点和核心指标的收敛表。"""
    summaries: list[dict[str, float]] = []
    for config in configs:
        result = solve_problem_2(first_position, first_bearing_deg, config)
        best = result.best_candidate
        summaries.append({
            "target_spacing": config.target_grid_spacing,
            "candidate_spacing": config.candidate_grid_spacing,
            "refined_spacing": config.refined_grid_spacing,
            "best_x": best.point.x,
            "best_y": best.point.y,
            "best_a": best.a,
            "best_b": best.b,
            "worst_radius": best.worst_radius,
            "clear_fraction": best.clear_fraction,
            "expected_time": best.expected_total_time,
        })
    return summaries


def _plot_geometry(ax, geometry: Region, **kwargs) -> None:
    from matplotlib.patches import Polygon as PolygonPatch

    if geometry.is_empty:
        return
    for ring in geometry.exterior_rings():
        coordinates = np.asarray(ring)
        ax.add_patch(PolygonPatch(coordinates, closed=True, **kwargs))


def plot_problem_2_result(
    result: Problem2Result,
    output_path: str | Path | None = None,
    show: bool = False,
):
    """绘制第一次可行域、候选点评分、保证接收区和推荐区域。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for preferred_font in (
        "PingFang SC",
        "Arial Unicode MS",
        "Heiti SC",
        "STHeiti",
        "Microsoft YaHei",
        "Noto Sans SC",
        "SimHei",
    ):
        if preferred_font in available_fonts:
            plt.rcParams["font.sans-serif"] = [preferred_font, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break

    figure, axis = plt.subplots(figsize=(10, 8))
    _plot_geometry(axis, result.first_feasible_region, facecolor="#76b7b2", alpha=0.4,
                   edgecolor="#24726c", linewidth=1.2, label="第一次可行域")
    _plot_geometry(axis, result.safe_candidate_region, facecolor="#59a14f", alpha=0.15,
                   edgecolor="#3a7d32", linewidth=1.0, label="保证接收区域")
    _plot_geometry(axis, result.recommended_region, facecolor="#e15759", alpha=0.25,
                   edgecolor="#b52f31", linewidth=1.3, label="推荐候选区域")

    x = [candidate.point.x for candidate in result.candidates]
    y = [candidate.point.y for candidate in result.candidates]
    weighted = result.config.optimization_mode == "robust" and result.config.signal_fraction_power > 0
    score = [adjusted_worst_radius(candidate, result.config) if weighted else candidate.worst_radius
             for candidate in result.candidates]
    scatter = axis.scatter(x, y, c=score, cmap="viridis_r", s=13, alpha=0.75)
    figure.colorbar(scatter, ax=axis, label="接收率调整评分（非实际半径）" if weighted else "最坏后验覆盖半径 / m")
    axis.scatter(
        [result.first_position.x], [result.first_position.y], marker="s", s=70,
        color="black", label="第一次检测点"
    )
    axis.scatter(
        [result.best_candidate.point.x], [result.best_candidate.point.y], marker="*",
        s=180, color="#d62728", edgecolor="white", linewidth=0.8, label="最优候选点"
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x / m")
    axis.set_ylabel("y / m")
    axis.set_title("问题2：第二检测点搜索结果")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left", framealpha=0.85)
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(Path(output_path), dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axis


def print_result_summary(result: Problem2Result) -> None:
    """打印便于核对和写作引用的结果摘要。"""
    best = result.best_candidate
    print("问题2第二检测点搜索摘要")
    print(f"第一次可行域面积：{result.first_feasible_region.area:.3f} m²")
    print(f"目标单元数：{len(result.target_cells)}")
    print(f"联合状态数：{len(result.states)}")
    print(f"候选点数：{len(result.candidates)}")
    print(f"最优点全局坐标：({best.point.x:.3f}, {best.point.y:.3f})")
    print(f"最优点局部坐标：(a={best.a:.3f}, b={best.b:.3f})")
    print(f"最坏后验覆盖半径：{best.worst_radius:.3f} m")
    if result.config.optimization_mode == "robust":
        print(f"接收率调整评分：{adjusted_worst_radius(best, result.config):.3f} (α={result.config.signal_fraction_power:g})")
    print(f"离散模型下可一次清除情形比例：{best.clear_fraction:.4f}")
    print(f"是否所有离散观测情形均可一次清除：{'是' if best.guaranteed_clear else '否'}")
    print(f"预计总时间（辅助指标）：{best.expected_total_time:.3f} s")
    print(f"离散模型下可探测样本权重：{best.signal_fraction:.4f}")
    print(f"是否位于保守保证接收区：{'是' if best.guaranteed_signal else '否'}")


def main() -> None:
    """支持正式高精度数值求解（默认）或快速演示。"""
    import argparse

    parser = argparse.ArgumentParser(description="CUMCM 2026 B题 问题2：第二检测点选择与候选区域求解")
    parser.add_argument("--demo", action="store_true", help="使用快速演示参数运行 (目标步长60m, 粗筛200m, 细筛50m, 种子4个)")
    parser.add_argument("--signal-fraction-power", type=float, default=None,
                        help="接收率惩罚指数；未指定时使用 Problem2Config 中的值")
    parser.add_argument("--allow-unguaranteed-signal", action="store_true",
                        help="关闭保证接收区域内部选点约束，用于软惩罚对照")
    parser.add_argument("--x", type=float, default=0.0, help="第一次检测点 x 坐标 (米，默认 0.0)")
    parser.add_argument("--y", type=float, default=0.0, help="第一次检测点 y 坐标 (米，默认 0.0)")
    parser.add_argument("--bearing", type=float, default=0.0, help="第一次测得的示向度 (度，默认 0.0)")
    parser.add_argument("--out", type=str, default=None, help="图片保存路径 (默认全功能为 P2_result.png, demo 为 P2_demo.png)")
    args = parser.parse_args()

    first_pos = (args.x, args.y)
    bearing = args.bearing

    if args.demo:
        config = Problem2Config(
            target_grid_spacing=60.0,
            candidate_grid_spacing=200.0,
            refined_grid_spacing=50.0,
            refinement_seed_count=4,
        )
        out_path = args.out or "P2_demo.png"
        print(">>> 当前运行模式：【快速演示模式】(--demo)。")
        print(">>> 网格参数：目标间距 60m, 粗筛间距 200m, 细筛间距 50m, 种子数 4\n")
    else:
        config = Problem2Config()
        out_path = args.out or "P2_result.png"
        print(f">>> 当前运行模式：【默认全功能高精度模式】检测点={first_pos}, 示向度={bearing}° ...")
        print(">>> 网格参数：目标间距 20m, 粗筛间距 100m, 细筛间距 25m, 种子数 8")
        print(">>> 正在进行全情景后验推断与双层网格优化（预计耗时 1~2 分钟）...")
        print(">>> 提示：如需秒级快速测试，可追加 --demo 参数\n")

    config = replace(config, require_guaranteed_signal=not args.allow_unguaranteed_signal)
    if args.signal_fraction_power is not None:
        config = replace(config, signal_fraction_power=args.signal_fraction_power)
    result = solve_problem_2(first_pos, bearing, config)
    print_result_summary(result)
    plot_problem_2_result(result, out_path)
    print(f"\n可视化图像已保存至: {out_path}")


if __name__ == "__main__":
    main()
