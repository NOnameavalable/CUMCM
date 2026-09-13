"""问题4的测点、探针和清除决策。"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Any, Iterable

from P2 import Problem2Result
from P3 import RouteTask, TaskPlanner, TaskType
from utils import Point, Region, minimum_enclosing_circle

from .belief import MixedCandidateEvaluation, MixedSourceBelief


@dataclass(frozen=True)
class ClearPlan:
    point: Point
    guaranteed: bool
    covered_probability: float


def plan_geometry_measurement(region: Region, measured_points: Iterable[Point], clear_radius: float) -> Point:
    """从最新区域中心及周边生成新测点；盲区测量后不会重复原点。"""
    if region.is_empty:
        raise RuntimeError("P1连续区域为空，观测或清除记录不一致。")
    circle = minimum_enclosing_circle(region.convex_hull.vertices)
    measured = list(measured_points)
    center = circle.center
    if all(center.distance_to(p) > 1.0 for p in measured):
        return center
    # 在当前区域尺度上从内到外换方向，避免退回首测候选列表。
    base = max(clear_radius, circle.radius * 0.5)
    for ring in range(1, len(measured) + 2):
        for i in range(16):
            angle = i * math.pi / 8
            point = Point(center.x + base * ring * math.cos(angle),
                          center.y + base * ring * math.sin(angle))
            if all(point.distance_to(p) > 1.0 for p in measured):
                return point
    raise RuntimeError("无法生成新的P1测点。")


def plan_adaptive_measurements(
    posterior: MixedSourceBelief,
    p2_result: Problem2Result,
    current_position: Point,
    measured_points: Iterable[Point],
    *,
    candidate_limit: int = 48,
) -> list[MixedCandidateEvaluation]:
    """复用P2候选几何，以混合源后验重新排序。"""
    measured = {(round(point.x, 6), round(point.y, 6)) for point in measured_points}
    points: list[Point] = []
    for candidate in p2_result.candidates:
        key = round(candidate.point.x, 6), round(candidate.point.y, 6)
        if key not in measured:
            points.append(candidate.point)
        if len(points) >= candidate_limit:
            break
    evaluations = posterior.evaluate_candidates(points, current_position)
    return sorted(
        evaluations,
        key=lambda item: (
            item.worst_radius,
            item.expected_radius,
            -item.signal_probability,
            item.travel_distance,
        ),
    )


def plan_probe(
    posterior: MixedSourceBelief,
    first_position: Point,
    first_bearing_deg: float,
    current_position: Point,
    measured_points: Iterable[Point],
) -> MixedCandidateEvaluation | None:
    """沿首测示向轴生成探针；探针只负责选点，不维护一维真假区间。"""
    marginals = posterior.target_marginals()
    if not marginals:
        return None
    rad = math.radians(first_bearing_deg)
    ux, uy = math.cos(rad), math.sin(rad)
    depths = sorted(
        (
            (point.x - first_position.x) * ux + (point.y - first_position.y) * uy,
            weight,
        )
        for point, weight in marginals
    )
    total = sum(weight for _, weight in depths)
    candidates: list[Point] = []
    for quantile in (0.25, 0.5, 0.75):
        threshold = total * quantile
        accumulated = 0.0
        for depth, weight in depths:
            accumulated += weight
            if accumulated >= threshold:
                candidates.append(
                    Point(first_position.x + depth * ux, first_position.y + depth * uy)
                )
                break
    measured = {(round(point.x, 6), round(point.y, 6)) for point in measured_points}
    candidates = [
        point for point in candidates
        if (round(point.x, 6), round(point.y, 6)) not in measured
    ]
    if not candidates:
        return None
    return min(
        posterior.evaluate_candidates(candidates, current_position),
        key=lambda item: (item.worst_radius, item.expected_radius, item.travel_distance),
    )


def plan_clear(
    posterior: MixedSourceBelief,
    failed_points: Iterable[Point],
    *,
    trial_probability: float = 0.72,
) -> ClearPlan | None:
    """选择保证清除点，或选择覆盖后验质量足够高的试射点。"""
    region = posterior.position_region
    if region.is_empty:
        return None
    failed_points = list(failed_points)
    failed = {(round(point.x, 6), round(point.y, 6)) for point in failed_points}
    vertices = region.convex_hull.vertices
    if vertices:
        circle = minimum_enclosing_circle(vertices)
        disk = Region.disk(circle.center, posterior.clear_radius)
        if circle.radius <= posterior.clear_radius + 1e-7 and disk.covers(region):
            key = round(circle.center.x, 6), round(circle.center.y, 6)
            if key not in failed:
                return ClearPlan(circle.center, True, 1.0)

    marginals = posterior.target_marginals()
    if not marginals:
        return None
    choices = sorted(marginals, key=lambda item: item[1], reverse=True)[:120]
    best_point = choices[0][0]
    best_probability = 0.0
    for candidate, _ in choices:
        key = round(candidate.x, 6), round(candidate.y, 6)
        if key in failed or any(
            candidate.distance_to(point) <= posterior.clear_radius + 1e-9
            for point in failed_points
        ):
            continue
        probability = sum(
            weight
            for point, weight in marginals
            if candidate.distance_to(point) <= posterior.clear_radius + 1e-9
        )
        if probability > best_probability:
            best_point, best_probability = candidate, probability
    if best_probability >= trial_probability:
        return ClearPlan(best_point, False, best_probability)
    return None


def optimize_shared_measurements(
    beliefs: dict[int, object],
    planner: TaskPlanner,
    current_position: Point,
    current_channel: int,
    *,
    time_budget_s: float,
) -> dict[str, Any]:
    """用各频道最新混合后验验证共享测点，再比较实际路线长度。"""
    started = time.monotonic()
    base_tasks = dict(planner.tasks)
    adaptable = [
        task
        for task in base_tasks.values()
        if task.task_type in (TaskType.SECOND_MEASURE, TaskType.FOLLOW_UP_MEASURE)
        and "探针" not in task.reason
        and not task.reason.startswith("P1")
        and task.point is not None
    ]
    if len(adaptable) < 2:
        return {"elapsed_s": 0.0, "saved_distance_m": 0.0, "assignments": {}}

    base_route = planner.plan_route(current_position, current_channel)
    previous = current_position
    base_distance = 0.0
    for task in base_route:
        if task.point is not None:
            base_distance += previous.distance_to(task.point)
            previous = task.point

    best_tasks = base_tasks
    best_distance = base_distance
    best_assignments: dict[int, Point] = {}
    shared_points = {
        (round(task.point.x, 8), round(task.point.y, 8)): task.point
        for task in adaptable
    }
    for shared_point in shared_points.values():
        if time.monotonic() - started >= time_budget_s:
            break
        trial = dict(base_tasks)
        assignments: dict[int, Point] = {}
        for task_id, task in base_tasks.items():
            if task not in adaptable or task.channel is None:
                continue
            belief = beliefs[task.channel]
            posterior = getattr(belief, "posterior", None)
            evaluations = getattr(belief, "candidate_evaluations", [])
            if posterior is None or not evaluations:
                continue
            baseline = evaluations[0]
            shared = posterior.evaluate_candidate(shared_point, current_position)
            if (
                shared.worst_radius <= baseline.worst_radius * 1.15 + 1e-9
                and shared.expected_radius <= baseline.expected_radius * 1.15 + 1e-9
                and shared.signal_probability + 1e-9 >= baseline.signal_probability * 0.8
            ):
                trial[task_id] = replace(task, point=shared_point, reason="混合后验共享测点")
                assignments[task.channel] = shared_point
        if len(assignments) < 2:
            continue
        planner.tasks = trial
        try:
            route = planner.plan_route(current_position, current_channel)
        finally:
            planner.tasks = base_tasks
        previous = current_position
        distance = 0.0
        for task in route:
            if task.point is not None:
                distance += previous.distance_to(task.point)
                previous = task.point
        if distance < best_distance - 1e-7:
            best_tasks = trial
            best_distance = distance
            best_assignments = assignments

    planner.tasks = best_tasks
    planner.plan_route(current_position, current_channel)
    elapsed = time.monotonic() - started
    return {
        "elapsed_s": elapsed,
        "saved_distance_m": max(0.0, base_distance - best_distance),
        "assignments": {
            channel: (point.x, point.y) for channel, point in best_assignments.items()
        },
    }
