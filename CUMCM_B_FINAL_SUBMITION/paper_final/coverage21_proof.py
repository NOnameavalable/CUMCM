"""Exact rational certificate for the 21-station comparison layout.

Default: verify the saved certificate, with Python standard library only.
--generate: rebuild the certificate (requires Shapely), then verify it.
The deployed 17-station controller is not imported or changed.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path

SCALE = 1_000_000
CERTIFICATE = Path(__file__).with_name('coverage21_certificate.json')


def coordinates():
    """Station positions are defined by rounding metre coordinates to 6 decimals."""
    stations = [(0, 0)]
    for count, radius in ((12, 1864), (8, 995)):
        stations.extend((round(radius * SCALE * math.cos(2 * math.pi * k / count)),
                         round(radius * SCALE * math.sin(2 * math.pi * k / count)))
                        for k in range(count))
    outer = 1800.1 / math.cos(math.pi / 12)
    polygon = [(round(outer * SCALE * math.cos(k * math.pi / 6)),
                round(outer * SCALE * math.sin(k * math.pi / 6))) for k in range(12)]
    return stations, polygon


def cross(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def clip(vertices, evaluate, positive):
    """Exact closed-half-plane clipping; all input coordinates are Fractions."""
    output = []
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        va, vb = evaluate(a), evaluate(b)
        inside_a = va >= 0 if positive else va <= 0
        inside_b = vb >= 0 if positive else vb <= 0
        if inside_a:
            output.append(a)
        if inside_a != inside_b:
            ratio = va / (va - vb)
            output.append(tuple(a[k] + ratio * (b[k] - a[k]) for k in (0, 1)))
    clean = []
    for point in output:
        if not clean or point != clean[-1]:
            clean.append(point)
    if len(clean) > 1 and clean[0] == clean[-1]:
        clean.pop()
    return clean


def witness_valid(vertices, stations, indices, radius_units):
    triangle = [stations[i] for i in indices]
    orientation = cross(*triangle)
    if not orientation:
        return False
    sign = 1 if orientation > 0 else -1
    for point in vertices:
        if any(sign * cross(triangle[i], triangle[(i + 1) % 3], point) < 0 for i in range(3)):
            return False
        if any((point[0] - q[0]) ** 2 + (point[1] - q[1]) ** 2 > radius_units ** 2
               for q in triangle):
            return False
    return True


def split_evaluator(vertices, node, stations):
    if 'line' in node:
        i, j = node['line']
        if not (isinstance(i, int) and isinstance(j, int) and 0 <= i < 21 and 0 <= j < 21 and i != j):
            raise ValueError('Invalid splitting station indices.')
        return lambda p: cross(stations[i], stations[j], p)
    axis = node['axis']
    if axis not in (0, 1):
        raise ValueError('Invalid bisection axis.')
    midpoint = (min(p[axis] for p in vertices) + max(p[axis] for p in vertices)) / 2
    return lambda p: p[axis] - midpoint


def verify(document):
    """No Shapely, floating geometry, random sampling, or numerical tolerance."""
    if document['units_per_metre'] != SCALE or document['target_radius'] != 1800:
        raise ValueError('Wrong target or coordinate units.')
    if document['certified_receive_radius'] != 999:
        raise ValueError('Unexpected certificate distance bound.')
    stations = [tuple(p) for p in document['stations']]
    polygon = [tuple(p) for p in document['enclosing_polygon']]
    expected, _ = coordinates()
    if stations != expected or len(polygon) != 12:
        raise ValueError('Certificate does not match the specified 21-station layout.')
    if any(type(x) is not int for p in stations + polygon for x in p):
        raise ValueError('Coordinates must be integer micrometres.')
    # Each directed edge defines an inward half-plane containing the full disk.
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        edge_sq = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
        origin_side = cross(a, b, (0, 0))
        if edge_sq == 0 or origin_side <= 0 or origin_side ** 2 < (1800 * SCALE) ** 2 * edge_sq:
            raise ValueError('Enclosing polygon fails exact circle containment.')
        if any(cross(a, b, p) < 0 for p in polygon):
            raise ValueError('Enclosing polygon is not convex and counterclockwise.')
    leaves = 0
    nodes = 0
    maximum_depth = 0

    def visit(vertices, node, depth=0):
        nonlocal leaves, nodes, maximum_depth
        nodes += 1
        if depth > 100 or len(vertices) < 3:
            raise ValueError('Invalid subdivision.')
        if 'w' in node:
            indices = node['w']
            if len(indices) != 3 or len(set(indices)) != 3 or any(type(i) is not int or not 0 <= i < 21 for i in indices):
                raise ValueError('Invalid witness triangle.')
            if not witness_valid(vertices, stations, indices, 999 * SCALE):
                raise ValueError('Exact vertex containment or distance inequality failed.')
            leaves += 1
            maximum_depth = max(maximum_depth, depth)
            return
        if len(node['children']) != 2 or ('line' in node) == ('axis' in node):
            raise ValueError('Every split must supply both children and one rule.')
        evaluate = split_evaluator(vertices, node, stations)
        values = [evaluate(p) for p in vertices]
        if not min(values) < 0 < max(values):
            raise ValueError('Split does not cut the parent polygon.')
        for positive, child in zip((False, True), node['children']):
            visit(clip(vertices, evaluate, positive), child, depth + 1)

    visit([(Fraction(x), Fraction(y)) for x, y in polygon], document['tree'])
    return {'verified': True, 'station_count': 21, 'target_radius_m': 1800,
            'certified_receive_radius_m': 999, 'leaf_count': leaves,
            'node_count': nodes, 'maximum_depth': maximum_depth,
            'arithmetic': 'exact integer and rational inequalities'}


def generate():
    """Floating geometry proposes witnesses; exact checks decide acceptance."""
    from shapely.geometry import Point, Polygon
    from shapely.strtree import STRtree
    stations, polygon = coordinates()
    disks = [Point(x / SCALE, y / SCALE).buffer(998, quad_segs=128) for x, y in stations]
    patches = []
    triples = []
    for indices in itertools.combinations(range(21), 3):
        patch = Polygon([(stations[i][0] / SCALE, stations[i][1] / SCALE) for i in indices])
        if patch.area < 1e-8:
            continue
        for i in indices:
            patch = patch.intersection(disks[i])
        if not patch.is_empty and patch.area > 1e-8:
            patches.append(patch)
            triples.append(indices)
    index = STRtree(patches)

    def build(vertices, depth=0):
        if depth > 100:
            raise RuntimeError('Subdivision did not obtain a certificate.')
        geometry = Polygon([(float(x / SCALE), float(y / SCALE)) for x, y in vertices])
        for k in index.query(geometry):
            if witness_valid(vertices, stations, triples[k], 999 * SCALE):
                return {'w': triples[k]}
        center = geometry.representative_point()
        candidates = [k for k in index.query(center) if patches[k].covers(center)]
        if not candidates:
            raise RuntimeError('No candidate witness; do not assert coverage.')
        for k in candidates:
            indices = triples[k]
            for i, j in zip(indices, indices[1:] + indices[:1]):
                node = {'line': [i, j]}
                evaluate = split_evaluator(vertices, node, stations)
                values = [evaluate(p) for p in vertices]
                if min(values) < 0 < max(values):
                    node['children'] = [build(clip(vertices, evaluate, sign), depth + 1)
                                        for sign in (False, True)]
                    return node
        spans = [max(p[k] for p in vertices) - min(p[k] for p in vertices) for k in (0, 1)]
        node = {'axis': max((0, 1), key=lambda k: spans[k])}
        evaluate = split_evaluator(vertices, node, stations)
        node['children'] = [build(clip(vertices, evaluate, sign), depth + 1) for sign in (False, True)]
        return node

    return {'units_per_metre': SCALE, 'target_radius': 1800, 'certified_receive_radius': 999,
            'stations': stations, 'enclosing_polygon': polygon,
            'tree': build([(Fraction(x), Fraction(y)) for x, y in polygon])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--certificate', type=Path, default=CERTIFICATE)
    arguments = parser.parse_args()
    if arguments.generate:
        data = generate()
        result = verify(data)
        arguments.certificate.write_text(json.dumps(data, separators=(',', ':')) + '\n')
    else:
        data = json.loads(arguments.certificate.read_text())
        result = verify(data)
    print(json.dumps(result, indent=2, ensure_ascii=False))
