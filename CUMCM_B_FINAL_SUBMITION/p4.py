"""第四题：混合源后验驱动的多频道搜索与清除控制。"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal
from P1 import solve_problem_1_geometry

from P2_cache import get_problem_2_result
from P3 import (
    ChannelBelief,
    ObservationRecord,
    Problem3Config,
    Problem3Controller,
    RouteTask,
    TaskPlanner,
    TaskType,
    main as run_cli,
)
from p4.belief import MixedCandidateEvaluation, MixedSourceBelief
from p4.planning import (
    optimize_shared_measurements as optimize_p4_shared_measurements,
    plan_adaptive_measurements,
    plan_clear,
    plan_probe,
    plan_geometry_measurement,
)
from p4.verify_coverage import generate_stations
from utils import Point, Region


@dataclass(frozen=True)
class Problem4PlanningConfig:
    """P4中需要通过实验调整的少量策略参数。"""

    orientation_step_deg: float = 10.0
    candidate_limit: int = 48
    no_signal_before_probe: int = 2
    trial_clear_probability: float = 0.72


@dataclass
class Problem4ChannelBelief(ChannelBelief):
    """P3通用频道事实，加上P4混合后验和搜索模式。"""

    posterior: MixedSourceBelief | None = field(default=None, repr=False)
    search_mode: Literal["adaptive", "probe"] = "adaptive"
    uninformative_count: int = 0
    candidate_evaluations: list[MixedCandidateEvaluation] = field(default_factory=list, repr=False)


def generate_coverage_points(config: Problem3Config) -> list[Point]:
    """生成现有17个巡检站；首次巡检布局不属于本次改动范围。"""
    stations, *_ = generate_stations(config.target_radius, config.receive_radius_min)
    return [Point(x, y) for x, y, _ in stations]


def refresh_channel_tasks_p4(
    belief: Problem4ChannelBelief,
    planner: TaskPlanner,
    config: Problem3Config,
    planning: Problem4PlanningConfig,
    current_position: Point,
) -> None:
    """把混合后验给出的决策转换成单个频道的下一任务。"""

    def task(task_type: TaskType, point: Point, reason: str) -> RouteTask:
        return RouteTask(
            task_id=planner.next_task_id(belief.channel, task_type),
            task_type=task_type,
            point=point,
            channel=belief.channel,
            channel_revision=belief.revision,
            reason=reason,
        )

    if belief.cleared:
        planner.remove_channel_tasks(belief.channel)
        return
    latest = belief.observations[-1] if belief.observations else None
    if latest is not None and latest.kind == "near":
        planner.replace_channel_tasks(
            belief.channel,
            [task(TaskType.CENTER_CLEAR, latest.position, "near结果位于清除半径内")],
        )
        return
    if not belief.direction_observations:
        coverage = [
            RouteTask(
                task_id=planner.next_task_id(belief.channel, TaskType.COVERAGE_MEASURE),
                task_type=TaskType.COVERAGE_MEASURE,
                point=point,
                channel=belief.channel,
                coverage_index=index,
                channel_revision=belief.revision,
                reason="17检测站巡检",
            )
            for index, point in enumerate(planner.coverage_points)
            if index not in belief.coverage_visited
        ]
        planner.replace_channel_tasks(belief.channel, coverage)
        return

    posterior = belief.posterior
    if posterior is None:
        raise RuntimeError(f"频道 {belief.channel} 已发现信号但尚未建立混合源后验。")
    directions = belief.direction_observations
    geometry_phase = len({(r.position.x, r.position.y) for r in directions}) >= 2
    if geometry_phase:
        geometry = solve_problem_1_geometry(
            [(r.position.x, r.position.y, float(r.bearing_deg)) for r in directions],
            error_deg=config.bearing_error_deg,
            target_radius=config.target_radius,
            receive_radius_max=config.receive_radius_max,
            near_radius=config.near_radius,
        )
        belief.p1_geometry = geometry
        region = geometry.polygon
        for point in belief.failed_clear_points:
            region = region.difference(Region.disk(point, config.clear_radius))
        posterior.position_region = region
        belief.candidate_evaluations.clear()
    belief.candidate_region = posterior.position_region
    clear = plan_clear(
        posterior,
        belief.failed_clear_points,
        trial_probability=float('inf') if geometry_phase else planning.trial_clear_probability,
    )
    if clear is not None:
        label = "后验区域保证覆盖" if clear.guaranteed else f"试射覆盖概率{clear.covered_probability:.1%}"
        planner.replace_channel_tasks(
            belief.channel,
            [task(TaskType.CENTER_CLEAR, clear.point, label)],
        )
        return

    measured = [record.position for record in belief.observations]
    if geometry_phase:
        point = plan_geometry_measurement(posterior.position_region, measured, config.clear_radius)
        planner.replace_channel_tasks(
            belief.channel,
            [task(TaskType.FOLLOW_UP_MEASURE, point, "P1连续区域后续测向")],
        )
        return
    first = belief.direction_observations[0]
    if belief.search_mode == "probe":
        assert first.bearing_deg is not None
        probe = plan_probe(
            posterior,
            first.position,
            first.bearing_deg,
            current_position,
            measured,
        )
        if probe is not None:
            belief.candidate_evaluations = [probe]
            planner.replace_channel_tasks(
                belief.channel,
                [task(TaskType.FOLLOW_UP_MEASURE, probe.point, "后验驱动探针测量")],
            )
            return
        belief.search_mode = "adaptive"

    result = belief.p2_result
    if result is None:
        raise RuntimeError(f"频道 {belief.channel} 缺少候选点基础几何。")
    evaluations = plan_adaptive_measurements(
        posterior,
        result,
        current_position,
        measured,
        candidate_limit=planning.candidate_limit,
    )
    belief.candidate_evaluations = evaluations
    if not evaluations:
        fallback = posterior.position_region.representative_point()
        planner.replace_channel_tasks(
            belief.channel,
            [task(TaskType.FOLLOW_UP_MEASURE, fallback, "后验网格耗尽后的区域代表点测量")],
        )
        return
    selected = evaluations[0]
    task_type = TaskType.SECOND_MEASURE if len(belief.direction_observations) == 1 else TaskType.FOLLOW_UP_MEASURE
    planner.replace_channel_tasks(
        belief.channel,
        [task(task_type, selected.point, "混合源后验自适应测量")],
    )
    belief.second_measure_point = selected.point


class Problem4Controller(Problem3Controller):
    """在P3通用执行循环上实现P4频道策略。"""

    def __init__(
        self,
        client,
        config: Problem3Config | None = None,
        planning_config: Problem4PlanningConfig | None = None,
    ) -> None:
        self.p4_config = planning_config or Problem4PlanningConfig()
        super().__init__(client, config)
        self.task_planner.coverage_points = generate_coverage_points(self.config)

    def create_channel_belief(self, channel: int) -> Problem4ChannelBelief:
        return Problem4ChannelBelief(channel)

    def _rebuild_posterior(self, belief: Problem4ChannelBelief) -> None:
        first = belief.direction_observations[0]
        assert first.bearing_deg is not None
        result = get_problem_2_result(
            (first.position.x, first.position.y),
            first.bearing_deg,
            replace(self.config.p2, bilateral_search=True),
        )
        belief.p2_result = result
        belief.p2_result_revision = belief.revision
        posterior = MixedSourceBelief.from_observations(
            result,
            belief.observations,
            bearing_error_deg=self.config.bearing_error_deg,
            near_radius=self.config.near_radius,
            clear_radius=self.config.clear_radius,
            orientation_step_deg=self.p4_config.orientation_step_deg,
        )
        for point in belief.failed_clear_points:
            posterior.exclude_clear_disk(point)
        belief.posterior = posterior
        belief.candidate_region = posterior.position_region

    def refresh_channel(self, belief: ChannelBelief) -> None:
        assert isinstance(belief, Problem4ChannelBelief)
        if belief.posterior is None and belief.direction_observations:
            self._rebuild_posterior(belief)
        refresh_channel_tasks_p4(
            belief,
            self.task_planner,
            self.config,
            self.p4_config,
            getattr(self.client, "position", Point(0.0, 0.0)),
        )

    def process_measurement(
        self,
        belief: ChannelBelief,
        task: RouteTask,
        record: ObservationRecord,
    ) -> None:
        assert isinstance(belief, Problem4ChannelBelief)
        if belief.posterior is None:
            if record.kind == "direction":
                self._rebuild_posterior(belief)
        elif not belief.posterior.update_measurement(record):
            self._rebuild_posterior(belief)

        if record.kind == "no_signal" and task.task_type is not TaskType.COVERAGE_MEASURE:
            belief.uninformative_count += 1
        elif record.kind in ("direction", "near"):
            belief.uninformative_count = 0
            belief.search_mode = "adaptive"
        if belief.uninformative_count >= self.p4_config.no_signal_before_probe:
            belief.search_mode = "probe"
        belief.candidate_evaluations.clear()
        belief.second_measure_completed = False
        belief.p2_result_revision = belief.revision

    def process_clear_failure(self, belief: ChannelBelief, task: RouteTask) -> None:
        assert isinstance(belief, Problem4ChannelBelief)
        if belief.posterior is None:
            self._rebuild_posterior(belief)
        assert belief.posterior is not None
        belief.posterior.exclude_clear_disk(task.point)
        belief.candidate_region = belief.posterior.position_region
        belief.candidate_evaluations.clear()
        if belief.posterior.position_region.is_empty:
            self._rebuild_posterior(belief)
            if belief.posterior is None or belief.posterior.position_region.is_empty:
                raise RuntimeError(f"频道 {belief.channel} 清除失败后后验与观测不一致。")

    def optimize_shared_measurements(self) -> dict[str, object]:
        return optimize_p4_shared_measurements(
            self.beliefs,
            self.task_planner,
            self.client.position,
            self.client.current_channel,
            time_budget_s=self.config.shared_planning_time_s,
        )


if __name__ == "__main__":
    run_cli(
        controller_type=Problem4Controller,
        coverage_generator=generate_coverage_points,
        problem_number=4,
    )
