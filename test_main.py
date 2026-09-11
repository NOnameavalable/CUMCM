import math

import pytest

from P1 import _diameter_circle_covers, solve_problem_1
from utils import Point, Polygon, Segment


def polygon_from_vertices(vertices):
    return Polygon([
        Segment(start, end)
        for start, end in zip(vertices, vertices[1:] + vertices[:1])
    ])


def test_problem_1_accepts_coordinate_and_bearing_tuples():
    diameter, can_cover = solve_problem_1([
        (-1000.0, 0.0, 0.0),
        (1000.0, 0.0, 180.0),
    ])
    assert diameter == pytest.approx(2000.0)
    assert can_cover is True


def test_additional_detection_sector_clips_initial_polygon():
    diameter, can_cover = solve_problem_1([
        (-2.0, 0.0, 0.0),
        (2.0, 0.0, 180.0),
        (0.0, -2.0, 90.0),
    ], error_deg=20.0)
    assert 0.0 < diameter <= 4.0
    assert isinstance(can_cover, bool)


def test_equilateral_triangle_is_not_covered_by_a_diameter_circle():
    height = math.sqrt(3.0)
    triangle = polygon_from_vertices([
        Point(-1.0, 0.0), Point(1.0, 0.0), Point(0.0, height)
    ])
    assert triangle.diameter() == pytest.approx(2.0)
    assert _diameter_circle_covers(triangle, triangle.diameter()) is False


@pytest.mark.parametrize(
    "vertices, expected_diameter, expected_cover",
    [
        ([Point(2, 3)], 0.0, True),
        ([Point(0, 0), Point(3, 4)], 5.0, True),
        ([Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2)], math.sqrt(8), True),
        ([Point(-1, 0), Point(1, 0), Point(0, math.sqrt(3))], 2.0, False),
    ],
)
def test_diameter_endpoints_and_single_circle_check(vertices, expected_diameter, expected_cover):
    polygon = polygon_from_vertices(vertices)
    diameter, first, second = polygon.diameter_with_endpoints()
    assert diameter == pytest.approx(expected_diameter)
    assert first.distance_to(second) == pytest.approx(diameter)
    assert polygon.diameter() == pytest.approx(diameter)
    assert _diameter_circle_covers(polygon, diameter, (first, second)) is expected_cover
    assert _diameter_circle_covers(polygon, diameter) is expected_cover


@pytest.mark.parametrize(
    "data",
    [
        [],
        [(0.0, 0.0, 0.0)],
        [(0.0, 0.0)],
        [(float("nan"), 0.0, 0.0), (1.0, 0.0, 180.0)],
    ],
)
def test_invalid_detection_data_is_rejected(data):
    with pytest.raises(ValueError):
        solve_problem_1(data)


def test_unbounded_initial_intersection_is_rejected():
    with pytest.raises(ValueError):
        solve_problem_1([
            (0.0, 0.0, 0.0),
            (0.0, 100.0, 0.0),
        ])
