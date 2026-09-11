import random

import pytest

from utils import DetectionSector, Point, Polygon, Ray, Segment, TargetArea


def polygon_from_vertices(vertices):
    return Polygon([
        Segment(start, end)
        for start, end in zip(vertices, vertices[1:] + vertices[:1])
    ])


def coordinates(polygon):
    return {
        (round(point.x, 8), round(point.y, 8))
        for point in polygon.vertices
    }


def shuffled_and_flipped(polygon):
    edges = [
        Segment(edge.end, edge.start) if index % 2 else edge
        for index, edge in enumerate(polygon.edges)
    ]
    random.Random(2026).shuffle(edges)
    return Polygon(edges)


@pytest.fixture
def target():
    return TargetArea()


@pytest.fixture
def square():
    return polygon_from_vertices([
        Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)
    ])


def test_unordered_edges_preserve_contains_vertices_and_diameter(square):
    unordered = shuffled_and_flipped(square)
    assert coordinates(unordered) == coordinates(square)
    assert unordered.contains(Point(0, 0))
    assert unordered.contains(Point(1, 0))
    assert not unordered.contains(Point(2, 0))
    assert unordered.diameter() == pytest.approx(2 ** 0.5 * 2)


def test_two_clips_include_first_cut_and_sector_origin(target, square):
    sector = DetectionSector(Ray(Point(0, 0), 315), Ray(Point(0, 0), 45))
    result = target.intersect_detection_area(square, sector)
    assert coordinates(result) == {(0.0, 0.0), (1.0, -1.0), (1.0, 1.0)}
    assert len(result.edges) == 3


def test_shuffling_and_flipping_do_not_change_clip(target, square):
    sector = DetectionSector(Ray(Point(0, 0), 315), Ray(Point(0, 0), 45))
    expected = target.intersect_detection_area(square, sector)
    actual = target.intersect_detection_area(shuffled_and_flipped(square), sector)
    assert coordinates(actual) == coordinates(expected)
    assert actual.diameter() == pytest.approx(expected.diameter())


def test_edge_with_both_endpoints_outside_can_cross_whole_sector(target):
    rectangle = polygon_from_vertices([
        Point(-1, -2), Point(1, -2), Point(1, 2), Point(-1, 2)
    ])
    sector = DetectionSector(Ray(Point(0, 0), 315), Ray(Point(0, 0), 45))
    result = target.intersect_detection_area(rectangle, sector)
    assert coordinates(result) == {(0.0, 0.0), (1.0, -1.0), (1.0, 1.0)}


def test_half_plane_three_endpoint_cases(target, square):
    boundary = Ray(Point(0, 0), 90)
    result = target._clip_polygon_by_half_plane(square, boundary, keep_left=False)
    assert coordinates(result) == {(0.0, -1.0), (1.0, -1.0), (1.0, 1.0), (0.0, 1.0)}
    assert len(result.edges) == 4


def test_tangent_empty_point_and_segment_degeneracies(target):
    square = polygon_from_vertices([
        Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2)
    ])
    tangent = target._clip_polygon_by_half_plane(
        square, Ray(Point(0, 0), 135), keep_left=True
    )
    assert coordinates(tangent) == {(0.0, 0.0)}
    assert tangent.diameter() == 0.0

    empty = target._clip_polygon_by_half_plane(
        square, Ray(Point(-1, 0), 90), keep_left=True
    )
    assert empty.edges == []

    line = Polygon([Segment(Point(-1, 0), Point(1, 0))])
    clipped_line = target._clip_polygon_by_half_plane(
        line, Ray(Point(0, 0), 90), keep_left=False
    )
    assert coordinates(clipped_line) == {(0.0, 0.0), (1.0, 0.0)}
    assert len(clipped_line.edges) == 1


def test_boundary_overlap_and_repeated_clip_are_stable(target, square):
    half = target._clip_polygon_by_half_plane(
        square, Ray(Point(0, 0), 90), keep_left=False
    )
    repeated = target._clip_polygon_by_half_plane(
        half, Ray(Point(0, 0), 90), keep_left=False
    )
    assert coordinates(repeated) == coordinates(half)
    assert len(repeated.edges) == len(half.edges)

    sector = DetectionSector(Ray(Point(0, 0), 315), Ray(Point(0, 0), 45))
    once = target.intersect_detection_area(square, sector)
    twice = target.intersect_detection_area(once, sector)
    assert coordinates(twice) == coordinates(once)
    assert len(twice.edges) == len(once.edges)


def test_invalid_sector_width_is_rejected(target, square):
    with pytest.raises(ValueError):
        target.intersect_detection_area(
            square, DetectionSector(Ray(Point(0, 0), 0), Ray(Point(0, 0), 0))
        )
    with pytest.raises(ValueError):
        target.intersect_detection_area(
            square, DetectionSector(Ray(Point(0, 0), 0), Ray(Point(0, 0), 181))
        )
