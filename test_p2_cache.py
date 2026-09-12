from dataclasses import replace
import math

import pytest

import P2_cache as cache
from P2 import Problem2Config, evaluate_problem_2_candidate
from utils import Point


@pytest.fixture
def config():
    return Problem2Config(target_grid_spacing=250, candidate_grid_spacing=500,
                          refined_grid_spacing=250, refinement_radius=300,
                          refinement_seed_count=2, receive_radius_sample_count=2,
                          error_samples_deg=(0.0,), observation_bin_deg=1,
                          circle_quad_segs=16, sector_arc_samples=17,
                          require_guaranteed_signal=False)


def test_complete_and_clipped_geometry(config):
    assert cache.sector_is_complete(Point(0, 0), 135, config)
    assert not cache.sector_is_complete(Point(1100, 0), 0, config)
    assert cache.sector_is_complete(Point(1100, 0), 180, config)
    assert not cache.sector_is_complete(Point(300, 0), 0, config)  # 相切保守实时计算


def test_disk_reuse_rotates_full_result_without_resolving(tmp_path, config, monkeypatch):
    template = cache.get_problem_2_result((0, 0), 0, config, cache_dir=tmp_path)
    assert template.result_source == "template_build"

    def forbidden(*args, **kwargs):
        pytest.fail("完整扇形命中磁盘模板后不能再次求解 P2")

    monkeypatch.setattr(cache, "solve_problem_2", forbidden)
    rotated = cache.get_problem_2_result((100, 50), 90, config, cache_dir=tmp_path)
    assert rotated.result_source == "template_reuse"
    for a, b in zip(template.candidates, rotated.candidates):
        assert b.point.x == pytest.approx(100 - a.point.y)
        assert b.point.y == pytest.approx(50 + a.point.x)
        assert b.worst_radius == a.worst_radius
        assert b.signal_fraction == a.signal_fraction
        for old, new in zip(a.posterior_cases, b.posterior_cases):
            assert new.clear_center.x == pytest.approx(100 - old.clear_center.y)
            if old.observation.bearing_deg is not None:
                assert new.observation.bearing_deg == pytest.approx((old.observation.bearing_deg + 90) % 360)
    assert rotated.first_feasible_region.area == pytest.approx(template.first_feasible_region.area)
    assert any(rotated.best_candidate is item for item in rotated.candidates)
    assert all(any(p is item for item in rotated.candidates) for p in rotated.pareto_candidates)
    metric = evaluate_problem_2_candidate(rotated, rotated.best_candidate.point)
    assert metric.signal_fraction == pytest.approx(rotated.best_candidate.signal_fraction)
    assert math.isfinite(metric.worst_radius)
    rotated.best_candidate.point.x = -9999
    again = cache.get_problem_2_result((0, 0), 0, config, cache_dir=tmp_path)
    assert again.best_candidate.point == template.best_candidate.point


def test_clipped_input_uses_live_solver(tmp_path, config, monkeypatch):
    original = cache.solve_problem_2
    calls = []

    def tracked(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(cache, "solve_problem_2", tracked)
    result = cache.get_problem_2_result((1100, 0), 0, config, cache_dir=tmp_path)
    assert result.result_source == "boundary_solve"
    assert calls[0][0][0] == (1100, 0)
    assert "_first_region" not in calls[0][1]
    assert not list(tmp_path.iterdir())


def test_cache_parameters_and_corruption(tmp_path, config):
    assert cache.template_cache_key(config) != cache.template_cache_key(replace(config, signal_fraction_power=2))
    cache.get_problem_2_result((0, 0), 0, config, cache_dir=tmp_path)
    next(tmp_path.glob("*.pickle")).write_bytes(b"invalid")
    with pytest.warns(RuntimeWarning, match="无效"):
        result = cache.get_problem_2_result((0, 0), 0, config, cache_dir=tmp_path)
    assert result.result_source == "template_build"
