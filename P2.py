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
from typing import Iterable, Literal, Sequence

import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from utils import Point


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
    refined_grid_spacing: float = 25.0
    refinement_radius: float = 160.0
    refinement_seed_count: int = 8
    candidate_margin: float = 300.0
    candidate_lateral_extent: float = 1000.0
    candidate_a_bounds: tuple[float, float] | None = None
    candidate_b_bounds: tuple[float, float] | None = None

    receive_radius_sample_count: int = 5
    error_samples_deg: tuple[float, ...] = (-1.0, 0.0, 1.0)
    observation_bin_deg: float = 0.25
    circle_quad_segs: int = 64
    sector_arc_samples: int = 65

    optimization_mode: OptimizationMode = "robust"
    candidate_region_relative_tolerance: float = 0.05
    candidate_region_absolute_tolerance: float = 2.0
    probability_tolerance: float = 0.02
    random_seed: int = 2026

    def __post_init__(self) -> None:
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
    geometry: BaseGeometry
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
    pareto_optimal: bool = False
    posterior_cases: list[PosteriorMetrics] = field(default_factory=list, repr=False)


@dataclass
class Problem2Result:
    config: Problem2Config
    first_position: Point
    first_bearing_deg: float
    first_feasible_region: BaseGeometry
    target_cells: list[TargetCell]
    states: list[StateSample]
    safe_candidate_region: BaseGeometry
    candidates: list[CandidateMetrics]
    best_candidate: CandidateMetrics
    pareto_candidates: list[CandidateMetrics]
    recommended_region: BaseGeometry


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
) -> BaseGeometry:
    """构造目标圆域、距离约束与第一次方向带的连续交集。"""
    target_disk = ShapelyPoint(0.0, 0.0).buffer(
        config.target_radius, quad_segs=config.circle_quad_segs
    )
    origin = ShapelyPoint(first_position.x, first_position.y)
    outer_disk = origin.buffer(config.receive_radius_max, quad_segs=config.circle_quad_segs)
    inner_disk = origin.buffer(config.near_radius, quad_segs=max(8, config.circle_quad_segs // 4))

    angles = np.linspace(
        first_bearing_deg - config.bearing_error_deg,
        first_bearing_deg + config.bearing_error_deg,
        config.sector_arc_samples,
    )
    arc = [
        (
            first_position.x + config.receive_radius_max * cos(radians(angle)),
            first_position.y + config.receive_radius_max * sin(radians(angle)),
        )
        for angle in angles
    ]
    sector = ShapelyPolygon([(first_position.x, first_position.y), *arc])
    region = target_disk.intersection(outer_disk).intersection(sector).difference(inner_disk)
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


def _hexagon(center_x: float, center_y: float, spacing: float) -> ShapelyPolygon:
    """返回三角晶格点的正六边形 Voronoi 单元。"""
    circumradius = spacing / sqrt(3.0)
    vertices = [
        (
            center_x + circumradius * cos(radians(30.0 + 60.0 * index)),
            center_y + circumradius * sin(radians(30.0 + 60.0 * index)),
        )
        for index in range(6)
    ]
    return ShapelyPolygon(vertices)


def _geometry_coordinates(geometry: BaseGeometry) -> Iterable[tuple[float, float]]:
    """枚举几何对象边界上的坐标，用于计算代表点覆盖误差。"""
    if geometry.geom_type == "Polygon":
        yield from geometry.exterior.coords
        for ring in geometry.interiors:
            yield from ring.coords
    elif geometry.geom_type == "MultiPolygon":
        for part in geometry.geoms:
            yield from _geometry_coordinates(part)
    elif hasattr(geometry, "coords"):
        yield from geometry.coords


def _sample_target_cells(
    first_region: BaseGeometry,
    config: Problem2Config,
) -> list[TargetCell]:
    """用裁剪六边形单元覆盖第一次连续可行域并按面积赋权。"""
    raw: list[tuple[Point, BaseGeometry, float, float]] = []
    for center_x, center_y in _generate_triangular_lattice(
        first_region.bounds, config.target_grid_spacing
    ):
        clipped = _hexagon(center_x, center_y, config.target_grid_spacing).intersection(first_region)
        if clipped.is_empty or clipped.area <= 1e-9:
            continue
        representative_shape = clipped.representative_point()
        representative = Point(float(representative_shape.x), float(representative_shape.y))
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
    first_region: BaseGeometry,
    first_position: Point,
    basis: tuple[np.ndarray, np.ndarray],
    config: Problem2Config,
) -> tuple[tuple[float, float], tuple[float, float]]:
    hull_coordinates = list(first_region.convex_hull.exterior.coords)
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
    b_bounds = config.candidate_b_bounds or (
        min(min(b_values) - config.candidate_margin, -config.candidate_lateral_extent),
        max(max(b_values) + config.candidate_margin, config.candidate_lateral_extent),
    )
    return a_bounds, b_bounds


def _inclusive_range(lower: float, upper: float, step: float) -> np.ndarray:
    count = max(1, int(np.floor((upper - lower) / step)))
    values = lower + np.arange(count + 1) * step
    if values[-1] < upper - 1e-9:
        values = np.append(values, upper)
    return values


def _generate_candidate_grid(
    first_region: BaseGeometry,
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
                point = _local_to_global(first_position, float(a), float(b), basis)
                if point.distance_to(first_position) <= 1e-9:
                    continue  # 同一位置重复检测的误差不会改变。
                key = (round(float(a) / step * 1_000_000), round(float(b) / step * 1_000_000))
                unique[key] = (point, float(a), float(b))
    return list(unique.values())


def _build_safe_candidate_region(
    first_region: BaseGeometry,
    config: Problem2Config,
) -> BaseGeometry:
    """近似计算保证距每个可行目标不超过最小接收半径的区域。"""
    hull = first_region.convex_hull
    coordinates = list(hull.exterior.coords)[:-1]
    safe: BaseGeometry | None = None
    for x, y in coordinates:
        disk = ShapelyPoint(x, y).buffer(
            config.receive_radius_min, quad_segs=max(16, config.circle_quad_segs // 2)
        )
        safe = disk if safe is None else safe.intersection(disk)
        if safe.is_empty:
            return GeometryCollection()
    return safe if safe is not None else GeometryCollection()


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


def _circle_from_two(first: Point, second: Point) -> EnclosingCircle:
    center = Point((first.x + second.x) / 2.0, (first.y + second.y) / 2.0)
    return EnclosingCircle(center, center.distance_to(first))


def _circle_from_three(first: Point, second: Point, third: Point) -> EnclosingCircle | None:
    determinant = 2.0 * (
        first.x * (second.y - third.y)
        + second.x * (third.y - first.y)
        + third.x * (first.y - second.y)
    )
    if abs(determinant) <= 1e-12:
        return None
    first_sq = first.x * first.x + first.y * first.y
    second_sq = second.x * second.x + second.y * second.y
    third_sq = third.x * third.x + third.y * third.y
    center = Point(
        (
            first_sq * (second.y - third.y)
            + second_sq * (third.y - first.y)
            + third_sq * (first.y - second.y)
        ) / determinant,
        (
            first_sq * (third.x - second.x)
            + second_sq * (first.x - third.x)
            + third_sq * (second.x - first.x)
        ) / determinant,
    )
    return EnclosingCircle(center, center.distance_to(first))


def _circle_contains(circle: EnclosingCircle, point: Point) -> bool:
    return circle.center.distance_to(point) <= circle.radius + max(1e-9, circle.radius * 1e-10)


def _minimum_enclosing_circle(
    points: Sequence[Point],
    random_seed: int = 2026,
) -> EnclosingCircle:
    """用固定随机顺序的增量算法求点集最小覆盖圆。"""
    if not points:
        raise ValueError("空点集不存在最小覆盖圆。")
    shuffled = list(points)
    np.random.default_rng(random_seed).shuffle(shuffled)
    circle: EnclosingCircle | None = None
    for i, first in enumerate(shuffled):
        if circle is not None and _circle_contains(circle, first):
            continue
        circle = EnclosingCircle(first, 0.0)
        for j, second in enumerate(shuffled[:i]):
            if _circle_contains(circle, second):
                continue
            circle = _circle_from_two(first, second)
            for third in shuffled[:j]:
                if _circle_contains(circle, third):
                    continue
                circumcircle = _circle_from_three(first, second, third)
                if circumcircle is not None:
                    circle = circumcircle
                else:
                    pairs = [(first, second), (first, third), (second, third)]
                    circle = max((_circle_from_two(*pair) for pair in pairs), key=lambda item: item.radius)
    assert circle is not None
    return circle


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
    safe_region: BaseGeometry,
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
    probe = ShapelyPoint(point.x, point.y)
    guaranteed_signal = not safe_region.is_empty and safe_region.covers(probe)
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
        posterior_cases=posterior_cases,
    )


def _candidate_sort_key(candidate: CandidateMetrics, mode: OptimizationMode) -> tuple[float, ...]:
    if mode == "robust":
        return (
            candidate.worst_radius,
            -candidate.clear_fraction,
            candidate.radius_q90,
            candidate.expected_radius,
            candidate.expected_total_time,
        )
    return (
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
    safe_region: BaseGeometry,
    config: Problem2Config,
) -> list[CandidateMetrics]:
    arrays = _state_arrays(states)
    results = [
        _evaluate_candidate(
            point, a, b, first_position, target_cells, states, arrays, safe_region, config
        )
        for point, a, b in raw_candidates
    ]
    return sorted(results, key=lambda item: _candidate_sort_key(item, config.optimization_mode))


def _refine_candidate_grid(
    coarse_results: Sequence[CandidateMetrics],
    first_region: BaseGeometry,
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
) -> BaseGeometry:
    """把相对最优的细网格点合并成稳定候选区域。"""
    best = candidates[0]
    radius_limit = (
        best.worst_radius * (1.0 + config.candidate_region_relative_tolerance)
        + config.candidate_region_absolute_tolerance
    )
    selected = [
        candidate
        for candidate in candidates
        if candidate.worst_radius <= radius_limit
        and candidate.clear_fraction >= best.clear_fraction - config.probability_tolerance
    ]
    if not selected:
        selected = [best]
    cell_radius = config.refined_grid_spacing / sqrt(2.0)
    return unary_union([
        ShapelyPoint(candidate.point.x, candidate.point.y).buffer(
            cell_radius, quad_segs=4
        )
        for candidate in selected
    ])


def solve_problem_2(
    first_position: Sequence[float],
    first_bearing_deg: float,
    config: Problem2Config | None = None,
) -> Problem2Result:
    """求第二检测点、Pareto 点集及近优候选区域。"""
    active_config = config or Problem2Config()
    first = _validate_inputs(first_position, first_bearing_deg)
    region = _build_first_feasible_region(first, first_bearing_deg, active_config)
    cells = _sample_target_cells(region, active_config)
    states = _expand_receive_radius_states(cells, first, active_config)
    safe_region = _build_safe_candidate_region(region, active_config)

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
        key=lambda item: _candidate_sort_key(item, active_config.optimization_mode),
    )
    pareto = _find_pareto_candidates(candidates)
    recommended_region = _extract_candidate_region(candidates, active_config)
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


def _plot_geometry(ax, geometry: BaseGeometry, **kwargs) -> None:
    from matplotlib.patches import Polygon as PolygonPatch

    if geometry.is_empty:
        return
    parts = geometry.geoms if isinstance(geometry, (MultiPolygon, GeometryCollection)) else [geometry]
    for part in parts:
        if part.geom_type != "Polygon":
            continue
        coordinates = np.asarray(part.exterior.coords)
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
    for preferred_font in ("Microsoft YaHei", "Noto Sans SC", "SimHei"):
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
    score = [candidate.worst_radius for candidate in result.candidates]
    scatter = axis.scatter(x, y, c=score, cmap="viridis_r", s=13, alpha=0.75)
    figure.colorbar(scatter, ax=axis, label="最坏后验覆盖半径 / m")
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
    axis.legend(loc="best")
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
    print(f"离散模型下可一次清除情形比例：{best.clear_fraction:.4f}")
    print(f"是否所有离散观测情形均可一次清除：{'是' if best.guaranteed_clear else '否'}")
    print(f"预计总时间（辅助指标）：{best.expected_total_time:.3f} s")
    print(f"是否位于保守保证接收区：{'是' if best.guaranteed_signal else '否'}")


def main() -> None:
    """运行一组明确标记为演示的参数。"""
    demo_config = Problem2Config(
        target_grid_spacing=60.0,
        candidate_grid_spacing=200.0,
        refined_grid_spacing=50.0,
        refinement_seed_count=4,
    )
    result = solve_problem_2((0.0, 0.0), 0.0, demo_config)
    print("注意：以下为程序连通性演示，不是题目正式最优结论。")
    print_result_summary(result)
    plot_problem_2_result(result, "P2_demo.png")


if __name__ == "__main__":
    main()
