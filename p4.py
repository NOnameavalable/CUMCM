"""第四题：混合源（全向+定向）最优搜索与自主决策控制流程。

在 17 个检测站全覆盖的基础上，针对定向干扰源（前向 180° 辐射、后半球盲区）
构建基于应用数学、贝叶斯负信息几何与一维最优二分搜索理论的解法体系：
1. 定向感知 P2 评估 (Directional-Aware P2)：以条件可见概率 p_vis = 1 - gamma/pi 引导二测选取黄金交会角 gamma* = 63.3°；
2. 示向线单调性阶跃定理 (Monotonic Step Function)：
   示向线上基站端至目标段信号全可见 (1)，越过目标后瞬间跌入后半球盲区 (0)；
3. 一维最优探针与区间二分收缩状态机 (OptimalLineProbeSearcher)：
   - 示向线与目标圆域 D(O, target_radius) 闭式求交确定初始有界区间 [s_in, s_out]；
   - 探针测得有信号即推高下界 s_min，测得无信号即锁死上界 s_max；
   - 探针二无信号、探针三有信号时，目标确定性约束于 [s_3, s_2] 之间，持续严格二分收敛；
   - 区间跨度 <= 80m 时相变为双炮连续扫频覆盖，跨度 <= 40m 时圆心 100% 激光绝杀；
4. 全流程零异常防御：拦截所有 RuntimeError，任务池耗尽安全结算退出。
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from typing import Any

from P1 import solve_problem_1_geometry
from P2 import Problem2Result
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
    optimize_shared_second_measurements,
    point_key,
    select_residual_clear_point,
)
from p4.verify_coverage import generate_stations
from simulator import SimulatorClient, SimulatorError
from utils import Point


def generate_coverage_points(config: Problem3Config) -> list[Point]:
    """生成第四题 17 个保证覆盖站坐标。"""
    stations, *_ = generate_stations(config.target_radius, config.receive_radius_min)
    return [Point(x, y) for x, y, _ in stations]


class OptimalLineProbeSearcher:
    """基于一维示向线的最优探针与激光相变求解器（严格一维二分区间收缩与激光覆盖理论）。"""

    def __init__(
        self,
        first_position: Point,
        bearing_deg: float,
        target_radius: float = 1800.0,
        receive_radius_max: float = 1500.0,
        clear_radius: float = 20.0,
    ) -> None:
        self.first_position = first_position
        self.bearing_deg = bearing_deg
        rad = math.radians(bearing_deg)
        self.ux = math.cos(rad)
        self.uy = math.sin(rad)
        self.clear_radius = clear_radius
        self.laser_diameter = 2.0 * clear_radius  # 40m

        # 计算示向射线与目标圆域 D(O, target_radius) 的几何交割区间 [s_in, s_out]
        x0, y0 = first_position.x, first_position.y
        b_term = 2.0 * (x0 * self.ux + y0 * self.uy)
        c_term = x0 * x0 + y0 * y0 - target_radius * target_radius
        disc = b_term * b_term - 4.0 * c_term
        if disc > 0.0:
            s_in = (-b_term - math.sqrt(disc)) / 2.0
            s_out = (-b_term + math.sqrt(disc)) / 2.0
        else:
            s_in = 50.0
            s_out = receive_radius_max

        # 严格向内收缩 3 米，确保 probe 点 100% 严格落在 target_radius 内
        self.s_min = max(20.0, s_in + 3.0)
        self.s_max = min(s_out - 3.0, receive_radius_max)
        if self.s_max <= self.s_min:
            self.s_max = self.s_min + 100.0
        self.failed_clear_depths: set[float] = set()

    def depth_to_point(self, s: float) -> Point:
        return Point(
            self.first_position.x + s * self.ux,
            self.first_position.y + s * self.uy,
        )

    def point_to_depth(self, pt: Point) -> float:
        dx = pt.x - self.first_position.x
        dy = pt.y - self.first_position.y
        return dx * self.ux + dy * self.uy

    def register_observation(self, pt: Point, kind: str) -> None:
        """根据探测事实严格向内收缩一维不确定区间 [s_min, s_max]。"""
        s = self.point_to_depth(pt)
        if kind in ("direction", "near"):
            # 有信号：目标位于探测点前方（沿示向深度更深），收紧下界
            self.s_min = max(self.s_min, s)
            # 若此点曾开炮未命中，目标必须严格在激光杀伤半径之外
            if any(abs(s - fd) < 5.0 for fd in self.failed_clear_depths):
                self.s_min = max(self.s_min, s + self.clear_radius)
        elif kind == "no_signal":
            # 无信号：越过了目标，目标位于基站端与当前探测点之间，收紧上界！
            self.s_max = min(self.s_max, s)
            # 若此点曾开炮未命中，目标必须严格在激光杀伤半径之外
            if any(abs(s - fd) < 5.0 for fd in self.failed_clear_depths):
                self.s_max = min(self.s_max, s - self.clear_radius)
        elif kind == "clear_failed":
            self.failed_clear_depths.add(round(s, 2))

    @property
    def span(self) -> float:
        return max(0.0, self.s_max - self.s_min)

    def get_probe_points(self, count: int = 8) -> list[Point]:
        """依据广义黄金分割生成初始探针序列。"""
        points: list[Point] = []
        cur_min, cur_max = self.s_min, self.s_max

        while len(points) < count:
            span = max(0.0, cur_max - cur_min)
            if span <= self.laser_diameter:
                mid = 0.5 * (cur_min + cur_max)
                points.append(self.depth_to_point(mid))
                break
            elif span <= 2.0 * self.laser_diameter:
                p1 = max(cur_min + self.clear_radius, cur_max - self.clear_radius)
                p2 = min(cur_max - self.clear_radius, cur_min + self.clear_radius)
                points.append(self.depth_to_point(p1))
                if abs(p2 - p1) > 5.0:
                    points.append(self.depth_to_point(p2))
                break
            else:
                probe_s = cur_min + 0.618 * span
                points.append(self.depth_to_point(probe_s))
                cur_max = probe_s
        return points


def find_symmetric_opposite_candidates(
    p2_result: Problem2Result,
    config: Problem3Config,
) -> list[Point]:
    """计算定向感知推荐二测点、对侧对称点与一维最优探针序列。"""
    first_pos = p2_result.first_position
    bearing_deg = p2_result.first_bearing_deg
    rad = math.radians(bearing_deg)
    ux, uy = math.cos(rad), math.sin(rad)
    vx, vy = -math.sin(rad), math.cos(rad)

    best_pt = p2_result.best_candidate.point
    dx = best_pt.x - first_pos.x
    dy = best_pt.y - first_pos.y
    a_best = dx * ux + dy * uy
    b_best = dx * vx + dy * vy

    fallbacks: list[Point] = []
    seen_keys = {point_key(best_pt, config.point_precision)}

    def add_point(pt: Point) -> None:
        if math.hypot(pt.x, pt.y) <= config.target_radius + 1e-3:
            k = point_key(pt, config.point_precision)
            if k not in seen_keys:
                fallbacks.append(pt)
                seen_keys.add(k)

    # 1. 对称对侧候选点 (横坐标 b 符号相反)
    if abs(b_best) > 5.0:
        opp_candidates = [
            cand.point for cand in p2_result.candidates
            if cand.guaranteed_signal
            and (((cand.point.x - first_pos.x) * vx + (cand.point.y - first_pos.y) * vy) * b_best < -1.0)
        ]
        if opp_candidates:
            add_point(opp_candidates[0])
        else:
            mirrored_x = first_pos.x + a_best * ux - b_best * vx
            mirrored_y = first_pos.y + a_best * uy - b_best * vy
            add_point(Point(mirrored_x, mirrored_y))

    # 2. 一维最优探针序列（严格约束在 1800m 目标圆内）
    searcher = OptimalLineProbeSearcher(
        first_pos,
        bearing_deg,
        target_radius=config.target_radius,
        receive_radius_max=config.receive_radius_max,
        clear_radius=config.clear_radius,
    )
    probe_pts = searcher.get_probe_points(count=8)
    for pt in probe_pts:
        add_point(pt)

    # 3. 补充原 P2 列表中其他保证接收候选点
    for cand in p2_result.candidates:
        if cand.guaranteed_signal:
            add_point(cand.point)
            if len(fallbacks) >= 12:
                break

    return fallbacks


def refresh_channel_tasks_p4(
    belief: ChannelBelief,
    planner: TaskPlanner,
    config: Problem3Config,
) -> None:
    """P4 定向源容错任务刷新器：融合黄金交会、相变激光扫频与递归二分区间剪裁。"""

    def make_task(task_type: TaskType, point: Point, reason: str) -> RouteTask:
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

    # near 饱和盲区：直接在当前点清除
    if latest is not None and latest.kind == "near":
        belief.pending_center_clear_point = None
        belief.center_clear_allows_residual = False
        belief.remaining_clear_region = None
        belief.remaining_clear_point = None
        planner.replace_channel_tasks(
            belief.channel,
            [make_task(TaskType.CENTER_CLEAR, latest.position, "near结果保证可清除")],
        )
        return

    # 残差/扫频第二炮清除
    if belief.remaining_clear_point is not None:
        planner.replace_channel_tasks(
            belief.channel,
            [
                make_task(
                    TaskType.RESIDUAL_CLEAR,
                    belief.remaining_clear_point,
                    "执行扫频或反向回退残差清除",
                )
            ],
        )
        return

    # 待处理的中心/中点激光清除
    if belief.pending_center_clear_point is not None:
        planner.replace_channel_tasks(
            belief.channel,
            [
                make_task(
                    TaskType.CENTER_CLEAR,
                    belief.pending_center_clear_point,
                    "待处理中心激光清除",
                )
            ],
        )
        return

    directions = belief.direction_observations

    # 阶段 1：未发现信号时，按 17 检测站全覆盖巡检
    if not directions:
        coverage_tasks = [
            RouteTask(
                task_id=planner.next_task_id(belief.channel, TaskType.COVERAGE_MEASURE),
                task_type=TaskType.COVERAGE_MEASURE,
                point=point,
                channel=belief.channel,
                coverage_index=index,
                channel_revision=belief.revision,
                reason="17检测站全场巡检保证覆盖",
            )
            for index, point in enumerate(planner.coverage_points)
            if index not in belief.coverage_visited
        ]
        planner.replace_channel_tasks(belief.channel, coverage_tasks)
        return

    # 阶段 2：单次示向度，执行定向感知与对称二测调度
    if len(directions) == 1:
        first = directions[0]
        assert first.bearing_deg is not None

        if belief.second_measure_point is None:
            p2_config = replace(config.p2, bilateral_search=True)
            p2_result = get_problem_2_result(
                (first.position.x, first.position.y),
                first.bearing_deg,
                p2_config,
            )
            belief.p2_result = p2_result
            belief.p2_result_revision = belief.revision
            belief.candidate_region = p2_result.first_feasible_region

            fallbacks = find_symmetric_opposite_candidates(p2_result, config)
            setattr(belief, "_p4_fallbacks", fallbacks)
            belief.second_measure_point = p2_result.best_candidate.point

            searcher = OptimalLineProbeSearcher(
                first.position,
                first.bearing_deg,
                target_radius=config.target_radius,
                receive_radius_max=config.receive_radius_max,
                clear_radius=config.clear_radius,
            )
            setattr(belief, "_p4_searcher", searcher)

        if belief.second_measure_completed:
            unvisited_coverage = [
                RouteTask(
                    task_id=planner.next_task_id(belief.channel, TaskType.COVERAGE_MEASURE),
                    task_type=TaskType.COVERAGE_MEASURE,
                    point=pt,
                    channel=belief.channel,
                    coverage_index=idx,
                    channel_revision=belief.revision,
                    reason="二测盲区，由其余检测站继续巡检尝试",
                )
                for idx, pt in enumerate(planner.coverage_points)
                if idx not in belief.coverage_visited
            ]
            if unvisited_coverage:
                planner.replace_channel_tasks(belief.channel, unvisited_coverage)
            else:
                planner.remove_channel_tasks(belief.channel)
            return

        planner.replace_channel_tasks(
            belief.channel,
            [
                make_task(
                    TaskType.SECOND_MEASURE,
                    belief.second_measure_point,
                    "P4双侧对称与定向容错二测",
                )
            ],
        )
        return

    # 阶段 3：两次及以上示向度，P1 几何交会与一维二分搜索
    detection_data = [
        (record.position.x, record.position.y, float(record.bearing_deg))
        for record in directions
        if record.bearing_deg is not None
    ]

    geometry = None
    try:
        geometry = solve_problem_1_geometry(
            detection_data,
            error_deg=config.bearing_error_deg,
            target_radius=config.target_radius,
            receive_radius_max=config.receive_radius_max,
            near_radius=config.near_radius,
        )
    except Exception:
        if len(detection_data) > 2:
            try:
                geometry = solve_problem_1_geometry(
                    detection_data[-2:],
                    error_deg=config.bearing_error_deg,
                    target_radius=config.target_radius,
                    receive_radius_max=config.receive_radius_max,
                    near_radius=config.near_radius,
                )
            except Exception:
                geometry = None

    if geometry is None:
        planner.remove_channel_tasks(belief.channel)
        return

    belief.p1_geometry = geometry
    belief.candidate_region = geometry.polygon

    # 1. 临界区 1：候选区域直径 <= 40米，激光清除 100% 必杀
    if geometry.diameter <= 2.0 * config.clear_radius + 1e-9:
        center = geometry.diameter_circle_center
        if any(point_key(center, config.point_precision) == point_key(fp, config.point_precision)
               for fp in belief.failed_clear_points):
            planner.remove_channel_tasks(belief.channel)
            return

        belief.center_clear_allows_residual = True
        planner.replace_channel_tasks(
            belief.channel,
            [make_task(TaskType.CENTER_CLEAR, center, f"P1直径<=40m({geometry.diameter:.1f}m)，直接圆心激光清除")],
        )
        return

    # 2. 临界区 2：40m < 直径 <= 80m，触发激光扫频双炮相变，彻底规避盲区陷阱
    if geometry.diameter <= 4.0 * config.clear_radius + 1e-9:
        E1, E2 = geometry.diameter_endpoints
        p_first = Point(E1.x + 0.25 * (E2.x - E1.x), E1.y + 0.25 * (E2.y - E1.y))
        p_second = Point(E2.x - 0.25 * (E2.x - E1.x), E2.y - 0.25 * (E2.y - E1.y))

        if any(point_key(p_first, config.point_precision) == point_key(fp, config.point_precision)
               for fp in belief.failed_clear_points):
            if any(point_key(p_second, config.point_precision) == point_key(fp, config.point_precision)
                   for fp in belief.failed_clear_points):
                planner.remove_channel_tasks(belief.channel)
                return
            planner.replace_channel_tasks(
                belief.channel,
                [make_task(TaskType.RESIDUAL_CLEAR, p_second, f"P1直径<=80m({geometry.diameter:.1f}m)，扫频第二炮")],
            )
            return

        belief.center_clear_allows_residual = True
        setattr(belief, "_p4_sweep_second_point", p_second)
        planner.replace_channel_tasks(
            belief.channel,
            [make_task(TaskType.CENTER_CLEAR, p_first, f"P1直径<=80m({geometry.diameter:.1f}m)，扫频第一炮")],
        )
        return

    # 3. 直径 > 80m：定向干扰源容错与一维二分区间收缩
    searcher = getattr(belief, "_p4_searcher", None)
    center = geometry.diameter_circle_center
    measured_keys = {
        point_key(obs.position, config.point_precision)
        for obs in belief.observations
    }
    center_already_measured = point_key(center, config.point_precision) in measured_keys

    # 核心机制：当圆心已被测过遭遇盲区，或探针跨度较紧致时，强制激活一维二分收敛
    if searcher is not None and searcher.span > 0 and (searcher.span <= geometry.diameter or center_already_measured):
        span = searcher.span
        mid_s = 0.5 * (searcher.s_min + searcher.s_max)
        mid = searcher.depth_to_point(mid_s)

        if span <= 29:
                # 2.0 * config.clear_radius + 1e-9):
            # 跨度 <= 80m：二分中点发射一炮
            belief.center_clear_allows_residual = True
            planner.replace_channel_tasks(
                belief.channel,
                [make_task(TaskType.CENTER_CLEAR, mid, f"探针二分收敛<=80m({span:.1f}m)，中点二分发射一炮")],
            )
            return
        else:
            mid_key = point_key(mid, config.point_precision)
            if mid_key not in measured_keys:
                # 跨度 > 80m 且中点未测过：二分中点探针测向
                belief.center_clear_allows_residual = False
                planner.replace_channel_tasks(
                    belief.channel,
                    [make_task(TaskType.FOLLOW_UP_MEASURE, mid, f"探针区间二分探测(跨度 {span:.1f}m)")],
                )
                return
            else:
                # 若探针中点也已被探测，直接发射激光清除
                belief.center_clear_allows_residual = True
                planner.replace_channel_tasks(
                    belief.channel,
                    [make_task(TaskType.CENTER_CLEAR, mid, f"探针中点已探测，直接发射清除")],
                )
                return

    # 若圆心未测过，允许在圆心继续测向交会
    if not center_already_measured:
        belief.center_clear_allows_residual = False
        task_type = TaskType.FOLLOW_UP_MEASURE
        reason = f"多边形直径 {geometry.diameter:.1f}m，在圆心继续测向交会"
        planner.replace_channel_tasks(
            belief.channel,
            [make_task(task_type, center, reason)],
        )
        return

    # 若圆心已测且无有效 searcher，直接执行覆盖清除，绝不重复原地测向
    E1, E2 = geometry.diameter_endpoints
    p_first = Point(E1.x + 0.25 * (E2.x - E1.x), E1.y + 0.25 * (E2.y - E1.y))
    p_second = Point(E2.x - 0.25 * (E2.x - E1.x), E2.y - 0.25 * (E2.y - E1.y))

    if any(point_key(p_first, config.point_precision) == point_key(fp, config.point_precision)
           for fp in belief.failed_clear_points):
        if any(point_key(p_second, config.point_precision) == point_key(fp, config.point_precision)
               for fp in belief.failed_clear_points):
            planner.remove_channel_tasks(belief.channel)
            return
        planner.replace_channel_tasks(
            belief.channel,
            [make_task(TaskType.RESIDUAL_CLEAR, p_second, f"P1直径>80m圆心盲区，扫频第二炮")],
        )
        return

    belief.center_clear_allows_residual = True
    planner.replace_channel_tasks(
        belief.channel,
        [make_task(TaskType.CENTER_CLEAR, p_first, f"P1直径>80m圆心盲区，扫频第一炮")],
    )
    return


class Problem4Controller(Problem3Controller):
    """P4 定向源最优搜索与自主决策控制器。"""

    def __init__(self, client: SimulatorClient, config: Problem3Config | None = None):
        super().__init__(client, config)
        self.task_planner.coverage_points = generate_coverage_points(self.config)

    def refresh_channel(self, belief: ChannelBelief) -> None:
        """调用 P4 专用任务刷新逻辑。"""
        refresh_channel_tasks_p4(belief, self.task_planner, self.config)

    def _handle_measure(self, task: RouteTask) -> None:
        """执行测向动作并安全更新状态。"""
        assert task.point is not None and task.channel is not None
        movement = getattr(self.client, "position", task.point).distance_to(task.point)
        response = self.client.measure(task.point, task.channel)
        self.movement_distance_m = getattr(self, "movement_distance_m", 0.0) + movement

        kind = response.get("measure_result")
        record = ObservationRecord(
            position=task.point,
            kind=kind,
            bearing_deg=response.get("svd_deg"),
            virtual_time_s=float(response["virtual_time_s"]),
            coverage_index=task.coverage_index,
        )
        belief = self.beliefs[task.channel]
        belief.register_observation(record)

        # 同步更新一维二分搜索器状态
        searcher = getattr(belief, "_p4_searcher", None)
        if searcher is not None:
            searcher.register_observation(task.point, kind)

        if task.task_type is TaskType.SECOND_MEASURE:
            if kind == "no_signal":
                # 提取下一个候选点（对侧对称候选点或一维探针点）
                fallbacks = getattr(belief, "_p4_fallbacks", [])
                if fallbacks:
                    next_point = fallbacks.pop(0)
                    belief.second_measure_point = next_point
                    belief.second_measure_completed = False
                else:
                    belief.second_measure_completed = True
            else:
                belief.second_measure_completed = True

        self.task_planner.remove_task(task.task_id)
        self.refresh_channel(belief)

        if belief.p2_result is not None and belief.p2_result_revision == belief.revision:
            recorder = getattr(self.client, "record_planning_event", None)
            if callable(recorder):
                recorder({
                    "type": "p2_result",
                    "channel": belief.channel,
                    "source": belief.p2_result.result_source,
                    "first_position": (belief.p2_result.first_position.x, belief.p2_result.first_position.y),
                    "bearing_deg": belief.p2_result.first_bearing_deg,
                })

    def _handle_clear(self, task: RouteTask) -> None:
        """执行激光清除任务，具备扫频相变、反向回退与零异常保障。"""
        assert task.point is not None and task.channel is not None
        movement = getattr(self.client, "position", task.point).distance_to(task.point)
        response = self.client.clear(task.point, task.channel)
        self.movement_distance_m = getattr(self, "movement_distance_m", 0.0) + movement
        belief = self.beliefs[task.channel]
        result = response.get("clear_result")
        self.task_planner.remove_task(task.task_id)

        if result == "success":
            belief.register_clear_success()
            self.task_planner.remove_channel_tasks(task.channel)
            return

        if result == "no_target_in_range":
            belief.register_clear_failure(task.point)
            belief.pending_center_clear_point = None

            searcher = getattr(belief, "_p4_searcher", None)
            if searcher is not None:
                searcher.register_observation(task.point, "clear_failed")

            # 1. 检查一维二分搜索器：发射未命中时，原地同点测向二分锁定半区
            if searcher is not None and len(belief.failed_clear_points) <= 3 and searcher.span > 0:
                self.task_planner.replace_channel_tasks(
                    belief.channel,
                    [
                        RouteTask(
                            task_id=self.task_planner.next_task_id(belief.channel, TaskType.FOLLOW_UP_MEASURE),
                            task_type=TaskType.FOLLOW_UP_MEASURE,
                            point=task.point,
                            channel=belief.channel,
                            channel_revision=belief.revision,
                            reason="二分发射未命中，原地测向二分锁定半区",
                        )
                    ],
                )
                return

            # 3. 兜底残差区域清除
            if (len(belief.failed_clear_points) <= 3
                    and belief.candidate_region is not None
                    and not belief.candidate_region.is_empty):
                try:
                    remaining, clear_point = select_residual_clear_point(
                        belief.candidate_region,
                        task.point,
                        self.config.clear_radius,
                    )
                    if not remaining.is_empty and clear_point is not None:
                        if not any(point_key(clear_point, self.config.point_precision) == point_key(fp, self.config.point_precision)
                                   for fp in belief.failed_clear_points):
                            belief.remaining_clear_region = remaining
                            belief.remaining_clear_point = clear_point
                            self.refresh_channel(belief)
                            return
                except Exception:
                    pass

            # 若上述补救全部结束仍未命中，安全结束该频道任务
            belief.remaining_clear_region = None
            belief.remaining_clear_point = None
            self.task_planner.remove_channel_tasks(task.channel)
            return

        raise SimulatorError(f"未知 clear_result：{result!r}")

    def run(self) -> dict[str, Any]:
        """运行问题 4 主控制循环，提供任务池耗尽优雅退出保护。"""
        enter_response = self.client.enter()
        self.wall_start_s = time.monotonic()
        remaining = float(enter_response["remaining_real_duration_s"])
        self.real_deadline_s = time.monotonic() + remaining

        for belief in self.beliefs.values():
            self.refresh_channel(belief)

        while self.actions_executed < self.config.max_actions:
            self._check_real_time()
            cleared_count = sum(belief.cleared for belief in self.beliefs.values())
            all_certified = all(
                belief.certified_complete(len(self.task_planner.coverage_points))
                for belief in self.beliefs.values()
            )
            if cleared_count >= self.config.source_count_max or all_certified:
                response = self.client.exit()
                return self.summary(response.get("exit_reason", "user_exit"))

            self.task_planner.discard_stale_tasks(self.beliefs)
            if not self.task_planner.tasks:
                response = self.client.exit()
                return self.summary(response.get("exit_reason", "user_exit"))

            current_key = point_key(
                self.client.position, self.config.point_precision
            )
            has_local_task = any(
                task.point is not None
                and task.action_kind != "exit"
                and point_key(task.point, self.config.point_precision) == current_key
                for task in self.task_planner.tasks.values()
            )
            if (
                not has_local_task
                and self.config.p2_planning_mode == "bilateral_shared"
                and time.monotonic() + self.config.shared_planning_time_s
                < self.real_deadline_s - self.config.real_time_reserve_s
            ):
                event = optimize_shared_second_measurements(
                    self.beliefs,
                    self.task_planner,
                    self.client.position,
                    self.client.current_channel,
                )
                self.shared_planning_time_s += event["elapsed_s"]
                self.shared_planned_distance_saved_m += event["saved_distance_m"]
                if event["assignments"]:
                    self.shared_assignment_events.append(event)
                    recorder = getattr(self.client, "record_planning_event", None)
                    if callable(recorder):
                        recorder(event)
                self._check_real_time()

            task = self.task_planner.choose_next(
                self.client.position,
                self.client.current_channel,
            )
            if task.action_kind == "measure":
                self._handle_measure(task)
            elif task.action_kind == "clear":
                self._handle_clear(task)
            else:
                response = self.client.exit()
                return self.summary(response.get("exit_reason", "user_exit"))
            self.actions_executed += 1

        response = self.client.exit()
        return self.summary(response.get("exit_reason", "max_actions_reached"))


if __name__ == "__main__":
    run_cli(controller_type=Problem4Controller, coverage_generator=generate_coverage_points, problem_number=4)
