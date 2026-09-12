from P2 import Problem2Config, _generate_candidate_grid
from utils import Point, Region


def test_default_candidate_grid_searches_both_sides():
    region = Region.disk(Point(300.0, 0.0), 100.0)
    config = Problem2Config(
        candidate_grid_spacing=100.0,
        candidate_margin=100.0,
        candidate_lateral_extent=200.0,
        bilateral_search=True,
    )

    candidates = _generate_candidate_grid(region, Point(0.0, 0.0), 0.0, config)
    lateral = {round(b) for _, _, b in candidates}

    assert any(value < 0 for value in lateral)
    assert any(value > 0 for value in lateral)


def test_explicit_asymmetric_lateral_bounds_are_preserved_after_rotation():
    region = Region.disk(Point(0.0, 300.0), 100.0)
    config = Problem2Config(
        candidate_grid_spacing=100.0,
        candidate_a_bounds=(100.0, 200.0),
        candidate_b_bounds=(-200.0, 100.0),
        bilateral_search=True,
    )

    candidates = _generate_candidate_grid(region, Point(0.0, 0.0), 90.0, config)

    assert {round(b) for _, _, b in candidates} == {-200, -100, 0, 100}
    # theta=90° 时局部左法向为全局负 x，验证没有在旋转后镜像复制。
    assert {round(point.x) for point, _, _ in candidates} == {-100, 0, 100, 200}


def test_legacy_single_side_mode_keeps_nonnegative_lateral_grid():
    region = Region.disk(Point(300.0, 0.0), 100.0)
    config = Problem2Config(
        candidate_grid_spacing=100.0,
        candidate_margin=100.0,
        candidate_lateral_extent=200.0,
        bilateral_search=False,
    )

    candidates = _generate_candidate_grid(region, Point(0.0, 0.0), 0.0, config)

    assert all(b >= 0.0 for _, _, b in candidates)
