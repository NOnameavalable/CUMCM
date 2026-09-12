from types import SimpleNamespace

import math

import pytest

import P3
from P1 import Problem1Geometry
from P3 import (
    ChannelBelief,
    ObservationRecord,
    Problem3Config,
    TaskPlanner,
    TaskType,
    SharedCandidate,
    optimize_shared_second_measurements,
    refresh_channel_tasks,
    select_residual_clear_point,
)
from utils import Point, Region


def test_coverage_centers_use_hexagon_edges_as_chords():
    config = Problem3Config()
    points = P3.generate_coverage_points(config)
    assert len(points) == 7
    assert points[0] == Point(0, 0)
    rho = 1800 * math.cos(math.pi / 6) - math.sqrt(1000**2 - 900**2)
    for index, center in enumerate(points[1:]):
        angle = index * math.pi / 3
        assert center.distance_to(points[0]) == pytest.approx(rho)
        for delta in (-math.pi / 6, math.pi / 6):
            endpoint = Point(1800 * math.cos(angle + delta),
                             1800 * math.sin(angle + delta))
            assert center.distance_to(endpoint) == pytest.approx(1000)


def test_new_seven_disks_cover_target_region():
    config = Problem3Config()
    points = P3.generate_coverage_points(config)
    # Include the central disk seam, outer boundary and hexagon vertices.
    for radius in (0, 999, 1000, 1001, 1100, 1300, 1500, 1700, 1800):
        for step in range(720):
            angle = step * math.pi / 360
            target = Point(radius * math.cos(angle), radius * math.sin(angle))
            assert min(target.distance_to(center) for center in points) <= 1000 + 1e-8


def test_chord_coverage_radius_limits():
    with pytest.raises(ValueError, match="无法"):
        P3.generate_coverage_points(Problem3Config(receive_radius_min=899))
    assert P3.generate_coverage_points(Problem3Config(
        target_radius=1000, receive_radius_min=1000)) == [Point(0, 0)]


def _route_task(name, x, channel=3, kind=TaskType.FOLLOW_UP_MEASURE):
    return P3.RouteTask(name, kind, Point(x, 0), channel)


def test_open_tsp_can_choose_farther_first_for_shorter_complete_route():
    planner = TaskPlanner(Problem3Config())
    for name, x in [("near", 1), ("left", -2), ("right", 3)]:
        planner.add_task(_route_task(name, x))
    route = planner.plan_route(Point(0, 0), 3)
    assert [task.task_id for task in route] == ["left", "near", "right"]
    # Open path is 7m; choosing nearest first needs 8m. No return leg.
    points = [Point(0, 0)] + [task.point for task in route]
    assert sum(a.distance_to(b) for a, b in zip(points, points[1:])) == 7


def test_same_position_clear_preserves_current_measurement_channel():
    planner = TaskPlanner(Problem3Config())
    planner.add_task(_route_task("clear8", 0, 8, TaskType.CENTER_CLEAR))
    planner.add_task(_route_task("measure2", 0, 2))
    planner.add_task(_route_task("measure3", 0, 3))
    route = planner.plan_route(Point(0, 0), 3)
    assert [task.task_id for task in route] == ["clear8", "measure3", "measure2"]


def test_tsp_replans_after_task_removal_and_addition():
    planner = TaskPlanner(Problem3Config())
    for name, x in [("near", 1), ("left", -2), ("right", 3)]:
        planner.add_task(_route_task(name, x))
    assert planner.choose_next(Point(0, 0), 3).task_id == "left"
    planner.remove_task("left")
    planner.remove_task("right")
    planner.add_task(_route_task("new", -3))
    route = planner.plan_route(Point(-2, 0), 3)
    assert [task.task_id for task in route] == ["new", "near"]


def test_initial_140_tasks_keep_all_channels_and_start_at_origin():
    config = Problem3Config()
    planner = TaskPlanner(config)
    for channel in range(1, 21):
        refresh_channel_tasks(ChannelBelief(channel), planner, config)
    route = planner.plan_route(Point(0, 0), 1)
    assert len(route) == len({task.task_id for task in route}) == 140
    assert all(task.point == Point(0, 0) for task in route[:20])
    assert route[0].channel == 1
    # One contiguous batch per location, although execution still takes one task.
    positions = [task.point for task in route]
    assert sum(a != b for a, b in zip(positions, positions[1:])) == 6


def test_exit_does_not_compete_with_tsp_tasks():
    planner = TaskPlanner(Problem3Config())
    planner.add_task(P3.RouteTask("exit", TaskType.EXIT, None, None))
    planner.add_task(_route_task("measure", 5))
    assert planner.choose_next(Point(0, 0), 3).task_id == "measure"
    planner.remove_task("measure")
    assert planner.choose_next(Point(0, 0), 3).task_type is TaskType.EXIT


class BatchClient:
    def __init__(self, near=False):
        self.position = Point(0, 0)
        self.current_channel = 1
        self.virtual_time_s = 0.0
        self.actions = []
        self.near = near

    def enter(self):
        return {"remaining_real_duration_s": 1200}

    def measure(self, point, channel):
        self.position = point
        self.current_channel = channel
        self.virtual_time_s += 5
        self.actions.append(("measure", point, channel))
        return {"measure_result": "near" if self.near else "no_signal",
                "virtual_time_s": self.virtual_time_s}

    def clear(self, point, channel):
        self.position = point
        self.actions.append(("clear", point, channel))
        self.virtual_time_s += 5
        return {"clear_result": "success", "virtual_time_s": self.virtual_time_s}

    def exit(self):
        return {"exit_reason": "user_exit"}


def test_coverage_batches_only_plan_when_leaving_a_position(monkeypatch):
    client = BatchClient()
    controller = P3.Problem3Controller(client)
    calls = []
    original = controller.task_planner.plan_route

    def counted(position, channel):
        calls.append(position)
        return original(position, channel)

    monkeypatch.setattr(controller.task_planner, "plan_route", counted)
    result = controller.run()
    assert result["actions_executed"] == 140
    # Origin needs no travel planning; only the six departures invoke TSP.
    assert len(calls) == 6
    for start in range(0, 140, 20):
        batch = client.actions[start:start + 20]
        assert len({(point.x, point.y) for _, point, _ in batch}) == 1
        assert {channel for _, _, channel in batch} == set(range(1, 21))


def test_batch_inserts_near_clear_and_checks_exit_before_next_channel(monkeypatch):
    client = BatchClient(near=True)
    controller = P3.Problem3Controller(client, Problem3Config(source_count_max=1))

    def unexpected_plan(*args):
        pytest.fail("原点批次不应调用TSP")

    monkeypatch.setattr(controller.task_planner, "plan_route", unexpected_plan)
    result = controller.run()
    assert [(kind, channel) for kind, _, channel in client.actions] == [
        ("measure", 1), ("clear", 1)]
    assert result["cleared_count"] == 1
    assert not any(task.channel == 1 for task in controller.task_planner.tasks.values())


def _direction(x: float, y: float, bearing: float) -> ObservationRecord:
    return ObservationRecord(Point(x, y), "direction", bearing, 0.0)


def _geometry(diameter: float, covers: bool, center: Point) -> Problem1Geometry:
    half = diameter / 2.0
    region = Region.from_vertices(
        [
            Point(center.x - half, center.y),
            Point(center.x, center.y + 1.0),
            Point(center.x + half, center.y),
        ]
    )
    return Problem1Geometry(
        polygon=region,
        diameter=diameter,
        diameter_endpoints=(
            Point(center.x - half, center.y),
            Point(center.x + half, center.y),
        ),
        diameter_circle_center=center,
        diameter_circle_covers=covers,
    )


def _only_task(planner: TaskPlanner):
    assert len(planner.tasks) == 1
    return next(iter(planner.tasks.values()))


def test_first_direction_uses_p2_and_adds_second_measure(monkeypatch):
    config = Problem3Config()
    planner = TaskPlanner(config)
    belief = ChannelBelief(channel=3)
    belief.register_observation(_direction(0.0, 0.0, 35.0))
    second = Point(400.0, 500.0)
    fallback = Point(250.0, 300.0)
    first_region = Region.disk(Point(0.0, 0.0), 10.0)

    monkeypatch.setattr(
        P3,
        "solve_problem_2",
        lambda *_args, **_kwargs: SimpleNamespace(
            best_candidate=SimpleNamespace(point=second),
            candidates=[
                SimpleNamespace(point=second, guaranteed_signal=False),
                SimpleNamespace(point=fallback, guaranteed_signal=True),
            ],
            first_feasible_region=first_region,
        ),
    )

    refresh_channel_tasks(belief, planner, config)
    task = _only_task(planner)

    assert task.task_type is TaskType.SECOND_MEASURE
    assert task.point == second
    assert belief.second_measure_point == second
    assert belief.second_measure_fallback_point == fallback
    assert belief.candidate_region is first_region


def test_second_measure_no_signal_uses_guaranteed_fallback(monkeypatch):
    config = Problem3Config()
    planner = TaskPlanner(config)
    belief = ChannelBelief(
        channel=4,
        second_measure_point=Point(400.0, 500.0),
        second_measure_fallback_point=Point(250.0, 300.0),
    )
    belief.register_observation(_direction(0.0, 0.0, 35.0))
    task = P3.RouteTask(
        task_id="second",
        task_type=TaskType.SECOND_MEASURE,
        point=belief.second_measure_point,
        channel=belief.channel,
        channel_revision=belief.revision,
        reason="共享候选点二次检测",
    )

    class FakeClient:
        def measure(self, _point, _channel):
            return {"measure_result": "no_signal", "virtual_time_s": 10.0}

    controller = SimpleNamespace(
        client=FakeClient(),
        config=config,
        beliefs={belief.channel: belief},
        task_planner=planner,
    )
    planner.add_task(task)
    P3.Problem3Controller._handle_measure(controller, task)

    fallback_task = _only_task(planner)
    assert fallback_task.task_type is TaskType.SECOND_MEASURE
    assert fallback_task.point == Point(250.0, 300.0)
    assert belief.second_measure_completed is False
    assert belief.shared_candidates_disabled is True


@pytest.mark.parametrize(
    ("diameter", "covers", "expected_type"),
    [
        (41.0, True, TaskType.FOLLOW_UP_MEASURE),
        (40.0, False, TaskType.CENTER_MEASURE),
        (40.0, True, TaskType.CENTER_CLEAR),
    ],
)
def test_p1_diameter_branches(
    monkeypatch, diameter: float, covers: bool, expected_type: TaskType
):
    config = Problem3Config()
    planner = TaskPlanner(config)
    belief = ChannelBelief(channel=8, second_measure_completed=True)
    belief.register_observation(_direction(-100.0, 0.0, 0.0))
    belief.register_observation(_direction(100.0, 0.0, 180.0))
    center = Point(12.0, -7.0)
    geometry = _geometry(diameter, covers, center)
    monkeypatch.setattr(P3, "solve_problem_1_geometry", lambda *_a, **_k: geometry)

    refresh_channel_tasks(belief, planner, config)
    task = _only_task(planner)

    assert task.task_type is expected_type
    assert task.point == center
    assert belief.p1_geometry is geometry
    assert belief.candidate_region is geometry.polygon
    assert belief.center_clear_allows_residual is (
        diameter <= 40.0 and not covers
    )


def test_uncovered_diameter_circle_measures_before_clearing(monkeypatch):
    config = Problem3Config()
    planner = TaskPlanner(config)
    belief = ChannelBelief(channel=5, second_measure_completed=True)
    belief.register_observation(_direction(-100.0, 0.0, 0.0))
    belief.register_observation(_direction(100.0, 0.0, 180.0))
    center = Point(0.0, 0.0)
    geometry = _geometry(30.0, False, center)
    monkeypatch.setattr(P3, "solve_problem_1_geometry", lambda *_a, **_k: geometry)

    refresh_channel_tasks(belief, planner, config)
    first_task = _only_task(planner)
    assert first_task.task_type is TaskType.CENTER_MEASURE

    class FakeClient:
        def measure(self, _point, _channel):
            return {
                "measure_result": "direction",
                "svd_deg": 90.0,
                "virtual_time_s": 10.0,
            }

    controller = SimpleNamespace(
        client=FakeClient(),
        config=config,
        beliefs={belief.channel: belief},
        task_planner=planner,
    )
    P3.Problem3Controller._handle_measure(controller, first_task)
    second_task = _only_task(planner)

    assert second_task.task_type is TaskType.CENTER_CLEAR
    assert second_task.point == center
    assert belief.pending_center_clear_point == center


def test_refresh_does_not_advance_pending_center_measure(monkeypatch):
    config = Problem3Config()
    planner = TaskPlanner(config)
    belief = ChannelBelief(channel=6, second_measure_completed=True)
    belief.register_observation(_direction(-100.0, 0.0, 0.0))
    belief.register_observation(_direction(100.0, 0.0, 180.0))
    center = Point(2.0, 3.0)
    monkeypatch.setattr(
        P3,
        "solve_problem_1_geometry",
        lambda *_a, **_k: _geometry(30.0, False, center),
    )

    refresh_channel_tasks(belief, planner, config)
    refresh_channel_tasks(belief, planner, config)

    task = _only_task(planner)
    assert task.task_type is TaskType.CENTER_MEASURE
    assert belief.pending_center_clear_point is None


def test_residual_point_is_selected_outside_failed_clear_disk():
    region = Region.from_vertices(
        [
            Point(-30.0, -10.0),
            Point(30.0, -10.0),
            Point(30.0, 10.0),
            Point(-30.0, 10.0),
        ]
    )
    remaining, point = select_residual_clear_point(
        region, Point(0.0, 0.0), 20.0
    )

    assert remaining.contains(point)
    assert point.distance_to(Point(0.0, 0.0)) >= 20.0 - 1e-7


def test_center_clear_failure_creates_one_residual_clear_task():
    config = Problem3Config()
    planner = TaskPlanner(config)
    region = Region.from_vertices(
        [
            Point(-30.0, -10.0),
            Point(30.0, -10.0),
            Point(30.0, 10.0),
            Point(-30.0, 10.0),
        ]
    )
    belief = ChannelBelief(
        channel=7,
        source_detected=True,
        candidate_region=region,
        pending_center_clear_point=Point(0.0, 0.0),
        center_clear_allows_residual=True,
    )
    task = P3.RouteTask(
        task_id="center-clear",
        task_type=TaskType.CENTER_CLEAR,
        point=Point(0.0, 0.0),
        channel=belief.channel,
        channel_revision=belief.revision,
    )

    class FakeClient:
        def clear(self, _point, _channel):
            return {"clear_result": "no_target_in_range", "virtual_time_s": 12.0}

    controller = SimpleNamespace(
        client=FakeClient(),
        config=config,
        beliefs={belief.channel: belief},
        task_planner=planner,
    )
    planner.add_task(task)
    P3.Problem3Controller._handle_clear(controller, task)

    residual_task = _only_task(planner)
    assert residual_task.task_type is TaskType.RESIDUAL_CLEAR
    assert belief.remaining_clear_region is not None
    assert belief.remaining_clear_region.contains(residual_task.point)
    assert residual_task.point.distance_to(Point(0.0, 0.0)) >= 20.0 - 1e-7


def test_controller_exits_after_reaching_source_count_max():
    config = Problem3Config(source_count_max=1)

    class FakeClient:
        def __init__(self):
            self.position = Point(0.0, 0.0)
            self.current_channel = 1
            self.virtual_time_s = 0.0
            self.exited = False

        def enter(self):
            return {"remaining_real_duration_s": 1200}

        def exit(self):
            self.exited = True
            return {"exit_reason": "user_exit"}

    client = FakeClient()
    controller = P3.Problem3Controller(client, config)
    controller.beliefs[1].register_clear_success()

    result = P3.run_controller(controller)

    assert client.exited is True
    assert result["cleared_count"] == 1


def test_near_creates_immediate_clear_task():
    config = Problem3Config()
    planner = TaskPlanner(config)
    belief = ChannelBelief(channel=11)
    point = Point(3.0, 4.0)
    belief.register_observation(ObservationRecord(point, "near", None, 5.0))

    refresh_channel_tasks(belief, planner, config)
    task = _only_task(planner)

    assert task.task_type is TaskType.CENTER_CLEAR
    assert task.point == point


def _shared_metric(point: Point, radius: float = 40.0):
    return SimpleNamespace(
        point=point,
        guaranteed_signal=True,
        worst_radius=radius,
    )


def test_shared_point_replaces_two_second_measurements_and_shortens_route(monkeypatch):
    config = Problem3Config(
        p2_planning_mode="bilateral_shared",
        shared_planning_time_s=2.0,
        shared_refinement_radius=1.0,
        shared_refined_spacing=1.0,
    )
    planner = TaskPlanner(config)
    beliefs = {}
    shared = Point(15.0, 0.0)
    for channel, independent in ((1, Point(10.0, 0.0)), (2, Point(20.0, 0.0))):
        belief = ChannelBelief(channel)
        belief.register_observation(_direction(0.0, 0.0, 0.0))
        old_metric = _shared_metric(independent, 20.0)
        shared_metric = _shared_metric(shared, 40.0)
        belief.p2_result = SimpleNamespace(
            candidates=[old_metric, shared_metric],
            best_candidate=old_metric,
            initial_worst_radius=100.0,
        )
        belief.p2_result_revision = belief.revision
        belief.second_measure_point = independent
        beliefs[channel] = belief
        planner.add_task(P3.RouteTask(
            f"second-{channel}", TaskType.SECOND_MEASURE, independent, channel,
            channel_revision=belief.revision,
        ))

    candidate = SharedCandidate(
        shared, {1, 2}, {1: _shared_metric(shared), 2: _shared_metric(shared)},
        {1: 1, 2: 1},
    )
    monkeypatch.setattr(P3, "build_shared_candidate_pool", lambda *_args: [candidate])
    monkeypatch.setattr(
        P3, "evaluate_problem_2_candidate", lambda _result, point: _shared_metric(point, 200.0)
    )
    event = optimize_shared_second_measurements(
        beliefs, planner, Point(0.0, 0.0), 1
    )

    second_tasks = [
        task for task in planner.tasks.values()
        if task.task_type is TaskType.SECOND_MEASURE
    ]
    assert len(second_tasks) == 2
    assert all(task.point == shared for task in second_tasks)
    assert event["saved_distance_m"] == pytest.approx(5.0)
    assert set(event["assignments"]) == {1, 2}


def test_shared_replacement_keeps_old_location_when_other_task_uses_it(monkeypatch):
    config = Problem3Config(
        p2_planning_mode="bilateral_shared",
        shared_planning_time_s=2.0,
        shared_refinement_radius=1.0,
        shared_refined_spacing=1.0,
    )
    planner = TaskPlanner(config)
    beliefs = {}
    shared = Point(15.0, 0.0)
    for channel, independent in ((1, Point(10.0, 0.0)), (2, Point(20.0, 0.0))):
        belief = ChannelBelief(channel)
        belief.register_observation(_direction(0.0, 0.0, 0.0))
        old_metric = _shared_metric(independent, 20.0)
        shared_metric = _shared_metric(shared, 40.0)
        belief.p2_result = SimpleNamespace(
            candidates=[old_metric, shared_metric], best_candidate=old_metric,
            initial_worst_radius=100.0,
        )
        belief.p2_result_revision = belief.revision
        beliefs[channel] = belief
        planner.add_task(P3.RouteTask(
            f"second-{channel}", TaskType.SECOND_MEASURE, independent, channel,
            channel_revision=belief.revision,
        ))
    planner.add_task(P3.RouteTask(
        "clear-at-old", TaskType.CENTER_CLEAR, Point(10.0, 0.0), 3
    ))
    candidate = SharedCandidate(
        shared, {1, 2}, {1: _shared_metric(shared), 2: _shared_metric(shared)},
        {1: 1, 2: 1},
    )
    monkeypatch.setattr(P3, "build_shared_candidate_pool", lambda *_args: [candidate])
    monkeypatch.setattr(
        P3, "evaluate_problem_2_candidate", lambda _result, point: _shared_metric(point, 200.0)
    )

    optimize_shared_second_measurements(beliefs, planner, Point(0.0, 0.0), 1)

    assert planner.tasks["clear-at-old"].point == Point(10.0, 0.0)
    assert sum(
        task.channel == 1 and task.task_type is TaskType.SECOND_MEASURE
        for task in planner.tasks.values()
    ) == 1


def test_stale_p2_cache_is_not_added_to_shared_pool():
    config = Problem3Config()
    planner = TaskPlanner(config)
    belief = ChannelBelief(1)
    belief.register_observation(_direction(0.0, 0.0, 0.0))
    belief.p2_result = SimpleNamespace(candidates=[_shared_metric(Point(10.0, 0.0))])
    belief.p2_result_revision = belief.revision
    belief.register_observation(ObservationRecord(
        Point(1.0, 0.0), "no_signal", None, 5.0
    ))

    assert P3.build_shared_candidate_pool(
        {1: belief}, planner, Point(0.0, 0.0), 1
    ) == []


def test_interpolated_shared_candidate_is_rejected_by_exact_evaluation(monkeypatch):
    config = Problem3Config(
        shared_planning_time_s=2.0,
        shared_refinement_radius=1.0,
        shared_refined_spacing=1.0,
    )
    planner = TaskPlanner(config)
    beliefs = {}
    proposed = Point(15.0, 0.0)
    for channel, independent in ((1, Point(10.0, 0.0)), (2, Point(20.0, 0.0))):
        belief = ChannelBelief(channel)
        belief.register_observation(_direction(0.0, 0.0, 0.0))
        old = _shared_metric(independent, 20.0)
        belief.p2_result = SimpleNamespace(
            candidates=[old], best_candidate=old, initial_worst_radius=100.0
        )
        belief.p2_result_revision = belief.revision
        beliefs[channel] = belief
        planner.add_task(P3.RouteTask(
            f"second-{channel}", TaskType.SECOND_MEASURE, independent, channel,
            channel_revision=belief.revision,
        ))
    coarse = SharedCandidate(
        proposed, {1, 2}, {1: _shared_metric(proposed), 2: _shared_metric(proposed)},
        {1: 1, 2: 1},
    )
    monkeypatch.setattr(P3, "build_shared_candidate_pool", lambda *_args: [coarse])
    monkeypatch.setattr(
        P3, "evaluate_problem_2_candidate", lambda _result, point: _shared_metric(point, 200.0)
    )

    event = optimize_shared_second_measurements(
        beliefs, planner, Point(0.0, 0.0), 1
    )

    assert event["assignments"] == {}
    assert planner.tasks["second-1"].point == Point(10.0, 0.0)
    assert planner.tasks["second-2"].point == Point(20.0, 0.0)
