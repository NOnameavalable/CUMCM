from types import SimpleNamespace

import pytest

import P3
from P1 import Problem1Geometry
from P3 import (
    ChannelBelief,
    ObservationRecord,
    Problem3Config,
    TaskPlanner,
    TaskType,
    refresh_channel_tasks,
    select_residual_clear_point,
)
from utils import Point, Region


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
