import math
import importlib.util
from pathlib import Path

from utils import Point


_SPEC = importlib.util.spec_from_file_location("p4_main", Path(__file__).with_name("p4.py"))
assert _SPEC is not None and _SPEC.loader is not None
_P4_MAIN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_P4_MAIN)
OptimalLineProbeSearcher = _P4_MAIN.OptimalLineProbeSearcher


FIRST_BEARING_DEG = 158.59
TARGET = Point(-778.79, 309.49)


def make_searcher() -> OptimalLineProbeSearcher:
    return OptimalLineProbeSearcher(
        Point(0.0, 0.0),
        FIRST_BEARING_DEG,
        target_radius=1800.0,
        receive_radius_max=1500.0,
        clear_radius=20.0,
    )


def test_far_side_no_signal_does_not_shrink_line_interval() -> None:
    searcher = make_searcher()
    old_max = searcher.s_max

    accepted = searcher.register_observation(Point(-512.81, 856.29), "no_signal")

    assert not accepted
    assert searcher.s_max == old_max
    assert searcher.last_rejection_reason is not None
    assert "横向偏离" in searcher.last_rejection_reason


def test_failed_midpoint_is_measured_then_returns_toward_pos1() -> None:
    searcher = make_searcher()
    pos1 = Point(-732.88, 287.36)
    pos2 = Point(-863.72, 347.91)
    pos3 = Point(-813.24, 325.12)

    # pos1 已清除失败，随后原地测得 direction：目标在 pos1 前方至少 20m。
    assert searcher.register_observation(pos1, "clear_failed")
    assert searcher.register_observation(pos1, "direction")

    # pos2 无信号：目标在 pos2 之前。
    assert searcher.register_observation(pos2, "no_signal")

    # pos3 清除失败并原地测得 no_signal：目标在 pos3 后方至少 20m。
    assert searcher.register_observation(pos3, "clear_failed")
    assert searcher.register_observation(pos3, "no_signal")

    next_depth = 0.5 * (searcher.s_min + searcher.s_max)
    next_point = searcher.depth_to_point(next_depth)
    pos1_depth = searcher.point_to_depth(pos1)
    pos3_depth = searcher.point_to_depth(pos3)

    assert searcher.interval_valid
    assert pos1_depth < next_depth < pos3_depth
    assert next_point.distance_to(TARGET) <= searcher.clear_radius


def test_contradictory_update_is_rejected_without_destroying_interval() -> None:
    searcher = make_searcher()
    searcher.s_min = 807.0
    searcher.s_max = 931.0

    accepted = searcher.register_observation(searcher.depth_to_point(790.0), "no_signal")

    assert not accepted
    assert math.isclose(searcher.s_min, 807.0)
    assert math.isclose(searcher.s_max, 931.0)
    assert searcher.interval_valid
    assert searcher.last_rejection_reason is not None
    assert "矛盾区间" in searcher.last_rejection_reason
