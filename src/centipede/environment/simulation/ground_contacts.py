# Warp kernels are checked by Warp's own compiler when first launched. The
# editor's type checker misreads their indexing, number types and vector
# maths, so the rules those false alarms come from are off in this file.
# pyright: reportIndexIssue=false, reportAssignmentType=false, reportGeneralTypeIssues=false
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportOperatorIssue=false

"""The contacts of the feet and leg links with a height field, on the GPU.

MuJoCo makes a height field's contacts column by column: each triangle of
its grid is a column reaching down to the field's base, and a shape gets one
contact for every column it overlaps, at the shortest way out of that column
(MuJoCo's ``mjc_ConvexHField``). MuJoCo Warp finds that way out with its
general convex check (GJK and EPA) in single precision, and at the body's
scale, feet 0.57 mm wide on columns 3 mm wide, the check fails: a foot whose
centre dips inside a steep column gets a contact millimetres deep, metres at
the far end of a 9 m field, and the solver throws the body (2026-10-24, the
study's Centipede ``archive/gpu_hfield_check/``). A sphere's or a capsule's
way out of a column has an exact answer, so the GPU computes these contacts
here, by MuJoCo's rule, with every length taken from the column's own corner
so that the field's size costs no precision.

MuJoCo Warp makes every other contact, the body shells' with the field
included, and is given a copy of the model in which the spheres and capsules
no longer meet the field. Its step then runs without its own collision stage
(``run_collision_detection``, its way of taking contacts made outside it);
the GPU backend collides before every step instead: MuJoCo Warp's contacts,
then these.
"""

import copy

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco_warp._src.collision_core import contact_params
from mujoco_warp._src.math import make_frame
from mujoco_warp._src.types import ContactType, vec5

# Nothing here is differentiated; Warp's derivative kernels would also need
# more parameter space than Pascal GPUs give a kernel.
wp.set_module_options({"enable_backward": False})

# MuJoCo's limit of contacts between two shapes (mjMAXCONPAIR).
MAX_CONTACTS_PER_SHAPE = mujoco.mjMAXCONPAIR
# A length below which a direction is taken as none, in m.
TINY_LENGTH = 1.0e-12
FAR = 1.0e30
ROUND_SHAPES = (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE)
# The shapes' kinds in the kernel.
SPHERE, CAPSULE, HULL = wp.constant(0), wp.constant(1), wp.constant(2)


@wp.func
def _over_triangle(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> bool:
    """Whether p lies over the triangle a, b, c: inside it seen from above."""
    d1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    d2 = (c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0])
    d3 = (a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0])
    negative = d1 < 0.0 or d2 < 0.0 or d3 < 0.0
    positive = d1 > 0.0 or d2 > 0.0 or d3 > 0.0
    return not (negative and positive)


@wp.func
def _side_normal(p: wp.vec3, q: wp.vec3, opposite: wp.vec3) -> wp.vec3:
    """The outward normal of the column's vertical side through edge p, q."""
    normal = wp.normalize(wp.vec3(q[1] - p[1], p[0] - q[0], 0.0))
    if wp.dot(normal, opposite - p) > 0.0:
        normal = -normal
    return normal


@wp.func
def _closest_on_segment(x: wp.vec3, p: wp.vec3, q: wp.vec3) -> wp.vec3:
    edge = q - p
    length2 = wp.dot(edge, edge)
    t = float(0.0)  # noqa: UP018 (Warp: a variable the branches change)
    if length2 > TINY_LENGTH * TINY_LENGTH:
        t = wp.clamp(wp.dot(x - p, edge) / length2, 0.0, 1.0)
    return p + t * edge


@wp.func
def _closest_between_segments(p1: wp.vec3, q1: wp.vec3, p2: wp.vec3, q2: wp.vec3):
    """The closest points of segments p1-q1 and p2-q2 (Ericson 2005, 5.1.9)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, r)
    tiny = TINY_LENGTH * TINY_LENGTH
    s = float(0.0)  # noqa: UP018 (Warp: a variable the branches change)
    t = float(0.0)  # noqa: UP018 (Warp: a variable the branches change)
    if a <= tiny and e <= tiny:
        s = 0.0
        t = 0.0
    elif a <= tiny:
        t = wp.clamp(f / e, 0.0, 1.0)
    else:
        c = wp.dot(d1, r)
        if e <= tiny:
            s = wp.clamp(-c / a, 0.0, 1.0)
        else:
            b = wp.dot(d1, d2)
            denominator = a * e - b * b
            if denominator > tiny * tiny:
                s = wp.clamp((b * f - c * e) / denominator, 0.0, 1.0)
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = wp.clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = wp.clamp((b - c) / a, 0.0, 1.0)
    return p1 + s * d1, p2 + t * d2


@wp.func
def _nearer(best: float, point: wp.vec3, x: wp.vec3, candidate: wp.vec3):
    """The candidate if it is nearer to x than the best so far."""
    distance2 = wp.length_sq(x - candidate)
    if distance2 < best:
        return distance2, candidate
    return best, point


@wp.func
def _side_projection(
    best: float,
    point: wp.vec3,
    x: wp.vec3,
    p: wp.vec3,
    q: wp.vec3,
    normal: wp.vec3,
    separation: float,
    base: float,
):
    """x's projection on the vertical side through edge p, q, if x is in
    front of that side and the projection falls on it."""
    if separation <= 0.0:
        return best, point
    projected = x - separation * normal
    edge = wp.vec3(q[0] - p[0], q[1] - p[1], 0.0)
    t = wp.dot(wp.vec3(projected[0] - p[0], projected[1] - p[1], 0.0), edge) / wp.dot(
        edge, edge
    )
    top = p[2] + t * (q[2] - p[2])
    if t >= 0.0 and t <= 1.0 and projected[2] >= base and projected[2] <= top:
        return _nearer(best, point, x, projected)
    return best, point


@wp.func
def _point_and_column(x: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3, base: float):
    """A point against a column: the triangle a, b, c on top, its heights in
    z, and vertical sides down to a flat bottom at height ``base``.

    Returns the point's signed distance from the column (inside, minus its
    depth below the nearest face), the unit direction out of the column
    toward the point, and the column's point nearest it (inside, the one on
    the nearest face).
    """
    top = wp.normalize(wp.cross(b - a, c - a))
    if top[2] < 0.0:
        top = -top
    bottom = wp.vec3(0.0, 0.0, -1.0)
    side_ab = _side_normal(a, b, c)
    side_bc = _side_normal(b, c, a)
    side_ca = _side_normal(c, a, b)
    above_top = wp.dot(top, x - a)
    below_bottom = base - x[2]
    beyond_ab = wp.dot(side_ab, x - a)
    beyond_bc = wp.dot(side_bc, x - b)
    beyond_ca = wp.dot(side_ca, x - c)
    if (
        above_top <= 0.0
        and below_bottom <= 0.0
        and beyond_ab <= 0.0
        and beyond_bc <= 0.0
        and beyond_ca <= 0.0
    ):
        # Inside: out through the nearest face.
        nearest = above_top
        normal = top
        if below_bottom > nearest:
            nearest = below_bottom
            normal = bottom
        if beyond_ab > nearest:
            nearest = beyond_ab
            normal = side_ab
        if beyond_bc > nearest:
            nearest = beyond_bc
            normal = side_bc
        if beyond_ca > nearest:
            nearest = beyond_ca
            normal = side_ca
        return nearest, normal, x - nearest * normal

    # Outside: the nearest of the faces it is in front of, and of the edges.
    best = float(FAR)  # noqa: UP018 (Warp: a variable the branches change)
    point = x
    if above_top > 0.0:
        projected = x - above_top * top
        if _over_triangle(projected, a, b, c):
            best, point = _nearer(best, point, x, projected)
    if below_bottom > 0.0 and _over_triangle(x, a, b, c):
        best, point = _nearer(best, point, x, wp.vec3(x[0], x[1], base))
    best, point = _side_projection(best, point, x, a, b, side_ab, beyond_ab, base)
    best, point = _side_projection(best, point, x, b, c, side_bc, beyond_bc, base)
    best, point = _side_projection(best, point, x, c, a, side_ca, beyond_ca, base)
    a_low = wp.vec3(a[0], a[1], base)
    b_low = wp.vec3(b[0], b[1], base)
    c_low = wp.vec3(c[0], c[1], base)
    best, point = _nearer(best, point, x, _closest_on_segment(x, a, b))
    best, point = _nearer(best, point, x, _closest_on_segment(x, b, c))
    best, point = _nearer(best, point, x, _closest_on_segment(x, c, a))
    best, point = _nearer(best, point, x, _closest_on_segment(x, a_low, b_low))
    best, point = _nearer(best, point, x, _closest_on_segment(x, b_low, c_low))
    best, point = _nearer(best, point, x, _closest_on_segment(x, c_low, a_low))
    best, point = _nearer(best, point, x, _closest_on_segment(x, a_low, a))
    best, point = _nearer(best, point, x, _closest_on_segment(x, b_low, b))
    best, point = _nearer(best, point, x, _closest_on_segment(x, c_low, c))
    distance = wp.sqrt(best)
    return distance, (x - point) / distance, point


@wp.func
def _overlap_along(
    axis: wp.vec3,
    s0: wp.vec3,
    s1: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    base: float,
    depth: float,
    normal: wp.vec3,
):
    """The separating-axis test along ``axis`` between segment s0-s1 and the
    column: whether the axis separates them, and the smallest push of the
    segment out of the column so far (its length and direction)."""
    length = wp.length(axis)
    if length < TINY_LENGTH:
        return False, depth, normal
    n = axis / length
    top_a = wp.dot(n, a)
    top_b = wp.dot(n, b)
    top_c = wp.dot(n, c)
    low_a = wp.dot(n, wp.vec3(a[0], a[1], base))
    low_b = wp.dot(n, wp.vec3(b[0], b[1], base))
    low_c = wp.dot(n, wp.vec3(c[0], c[1], base))
    column_min = wp.min(
        wp.min(wp.min(top_a, top_b), wp.min(top_c, low_a)), wp.min(low_b, low_c)
    )
    column_max = wp.max(
        wp.max(wp.max(top_a, top_b), wp.max(top_c, low_a)), wp.max(low_b, low_c)
    )
    segment_min = wp.min(wp.dot(n, s0), wp.dot(n, s1))
    segment_max = wp.max(wp.dot(n, s0), wp.dot(n, s1))
    if segment_max < column_min or column_max < segment_min:
        return True, depth, normal
    if column_max - segment_min < depth:
        depth = column_max - segment_min
        normal = n
    if segment_max - column_min < depth:
        depth = segment_max - column_min
        normal = -n
    return False, depth, normal


@wp.func
def _clip_by_face(
    s0: wp.vec3,
    u: wp.vec3,
    normal: wp.vec3,
    point: wp.vec3,
    first: float,
    last: float,
):
    """The part [first, last] of segment s0 + t u still inside the face
    through ``point`` with outward ``normal``."""
    start = wp.dot(normal, s0 - point)
    end = wp.dot(normal, s0 + u - point)
    if start > 0.0 and end <= 0.0:
        first = wp.max(first, start / (start - end))
    elif end > 0.0 and start <= 0.0:
        last = wp.min(last, start / (start - end))
    return first, last


@wp.func
def _segment_and_column(
    s0: wp.vec3, s1: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3, base: float
):
    """A segment against a column, as ``_point_and_column``: the signed
    distance, the unit direction out of the column toward the segment, the
    segment's point and the column's point that the contact joins.

    A segment crossing the column leaves it by the shortest push among the
    directions that can separate a segment from a column: the column's face
    normals and each of its edges crossed with the segment (the
    separating-axis test, exact for convex shapes). The contact is at the
    deepest point of the segment's part inside the column: a segment nearly
    parallel to the face it leaves by is as deep all along, and its far end
    may lie outside the column.
    """
    u = s1 - s0
    a_low = wp.vec3(a[0], a[1], base)
    b_low = wp.vec3(b[0], b[1], base)
    c_low = wp.vec3(c[0], c[1], base)
    depth = float(FAR)  # noqa: UP018 (Warp: a variable the branches change)
    normal = wp.vec3(0.0, 0.0, 1.0)
    apart = bool(False)  # noqa: UP018 (Warp: a variable the branches change)
    for axis_index in range(12):
        axis = wp.vec3(0.0, 0.0, 1.0)
        if axis_index == 0:
            axis = wp.cross(b - a, c - a)
        elif axis_index == 1:
            axis = wp.vec3(b[1] - a[1], a[0] - b[0], 0.0)
        elif axis_index == 2:
            axis = wp.vec3(c[1] - b[1], b[0] - c[0], 0.0)
        elif axis_index == 3:
            axis = wp.vec3(a[1] - c[1], c[0] - a[0], 0.0)
        elif axis_index == 4:
            axis = wp.vec3(0.0, 0.0, 1.0)
        elif axis_index == 5:
            axis = wp.cross(u, b - a)
        elif axis_index == 6:
            axis = wp.cross(u, c - b)
        elif axis_index == 7:
            axis = wp.cross(u, a - c)
        elif axis_index == 8:
            axis = wp.cross(u, b_low - a_low)
        elif axis_index == 9:
            axis = wp.cross(u, c_low - b_low)
        elif axis_index == 10:
            axis = wp.cross(u, a_low - c_low)
        else:
            axis = wp.cross(u, wp.vec3(0.0, 0.0, 1.0))
        if not apart:
            apart, depth, normal = _overlap_along(
                axis, s0, s1, a, b, c, base, depth, normal
            )
    if not apart:
        top = wp.normalize(wp.cross(b - a, c - a))
        if top[2] < 0.0:
            top = -top
        first, last = _clip_by_face(s0, u, top, a, 0.0, 1.0)
        first, last = _clip_by_face(s0, u, wp.vec3(0.0, 0.0, -1.0), a_low, first, last)
        first, last = _clip_by_face(s0, u, _side_normal(a, b, c), a, first, last)
        first, last = _clip_by_face(s0, u, _side_normal(b, c, a), b, first, last)
        first, last = _clip_by_face(s0, u, _side_normal(c, a, b), c, first, last)
        if first > last:
            first = 0.5 * (first + last)
            last = first
        inside0 = s0 + first * u
        inside1 = s0 + last * u
        lead0 = wp.dot(normal, inside0)
        lead1 = wp.dot(normal, inside1)
        deepest = 0.5 * (inside0 + inside1)
        if lead0 < lead1 - TINY_LENGTH:
            deepest = inside0
        elif lead1 < lead0 - TINY_LENGTH:
            deepest = inside1
        return -depth, normal, deepest, deepest + depth * normal

    # Apart: the nearest of the ends against the column and the segment
    # against the column's edges.
    distance, out, column_point = _point_and_column(s0, a, b, c, base)
    segment_point = s0
    distance1, out1, column_point1 = _point_and_column(s1, a, b, c, base)
    if distance1 < distance:
        distance = distance1
        out = out1
        column_point = column_point1
        segment_point = s1
    for edge_index in range(9):
        p = a
        q = b
        if edge_index == 1:
            p = b
            q = c
        elif edge_index == 2:
            p = c
            q = a
        elif edge_index == 3:
            p = a_low
            q = b_low
        elif edge_index == 4:
            p = b_low
            q = c_low
        elif edge_index == 5:
            p = c_low
            q = a_low
        elif edge_index == 6:
            p = a_low
            q = a
        elif edge_index == 7:
            p = b_low
            q = b
        elif edge_index == 8:
            p = c_low
            q = c
        on_segment, on_edge = _closest_between_segments(s0, s1, p, q)
        gap = wp.length(on_segment - on_edge)
        if gap < distance and gap > TINY_LENGTH:
            distance = gap
            out = (on_segment - on_edge) / gap
            column_point = on_edge
            segment_point = on_segment
    return distance, out, segment_point, column_point


@wp.func
def _column_corner(
    index: int,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    a_low: wp.vec3,
    b_low: wp.vec3,
    c_low: wp.vec3,
) -> wp.vec3:
    corner = a
    if index == 1:
        corner = b
    elif index == 2:
        corner = c
    elif index == 3:
        corner = a_low
    elif index == 4:
        corner = b_low
    elif index == 5:
        corner = c_low
    return corner


@wp.func
def _hull_overlap_along(
    axis: wp.vec3,
    hull_vertices: wp.array1d[wp.vec3],
    vertex_span: wp.vec2i,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    a_low: wp.vec3,
    b_low: wp.vec3,
    c_low: wp.vec3,
    depth: float,
    normal: wp.vec3,
):
    """``_overlap_along`` for a convex hull: the hull's vertices against the
    column's six corners, all in the hull's frame."""
    length = wp.length(axis)
    if length < TINY_LENGTH:
        return False, depth, normal
    n = axis / length
    hull_min = float(FAR)  # noqa: UP018 (Warp: a variable the loop changes)
    hull_max = float(-FAR)  # noqa: UP018 (Warp: a variable the loop changes)
    for index in range(vertex_span[0], vertex_span[0] + vertex_span[1]):
        along = wp.dot(n, hull_vertices[index])
        hull_min = wp.min(hull_min, along)
        hull_max = wp.max(hull_max, along)
    column_min = float(FAR)  # noqa: UP018 (Warp: a variable the loop changes)
    column_max = float(-FAR)  # noqa: UP018 (Warp: a variable the loop changes)
    for corner in range(6):
        along = wp.dot(n, _column_corner(corner, a, b, c, a_low, b_low, c_low))
        column_min = wp.min(column_min, along)
        column_max = wp.max(column_max, along)
    if hull_max < column_min or column_max < hull_min:
        return True, depth, normal
    if column_max - hull_min < depth:
        depth = column_max - hull_min
        normal = n
    if hull_max - column_min < depth:
        depth = hull_max - column_min
        normal = -n
    return False, depth, normal


@wp.func
def _hull_and_column(
    centre: wp.vec3,
    rotation: wp.mat33,
    column_a: wp.vec3,
    column_b: wp.vec3,
    column_c: wp.vec3,
    base: float,
    hull_vertices: wp.array1d[wp.vec3],
    vertex_span: wp.vec2i,
    hull_normals: wp.array1d[wp.vec3],
    normal_span: wp.vec2i,
    hull_edges: wp.array1d[wp.vec3],
    edge_span: wp.vec2i,
):
    """A convex hull, placed at ``centre`` and turned by ``rotation`` (its
    axes in the column's frame), against a column, as ``_segment_and_column``
    for a hull that counts only when it overlaps the column: the distance is
    then minus the shortest push out, among the column's and the hull's face
    normals and each hull edge crossed with each column edge (the
    separating-axis test), and the contact joins the two shapes' deepest
    features along that push. Apart, the distance is FAR."""
    to_hull = wp.transpose(rotation)
    a = to_hull @ (column_a - centre)
    b = to_hull @ (column_b - centre)
    c = to_hull @ (column_c - centre)
    a_low = to_hull @ (wp.vec3(column_a[0], column_a[1], base) - centre)
    b_low = to_hull @ (wp.vec3(column_b[0], column_b[1], base) - centre)
    c_low = to_hull @ (wp.vec3(column_c[0], column_c[1], base) - centre)
    depth = float(FAR)  # noqa: UP018 (Warp: a variable the branches change)
    normal = wp.vec3(0.0, 0.0, 1.0)
    apart = bool(False)  # noqa: UP018 (Warp: a variable the branches change)
    for face in range(5):
        axis = wp.cross(b - a, c - a)
        if face == 1:
            axis = to_hull @ wp.vec3(0.0, 0.0, 1.0)
        elif face == 2:
            axis = wp.cross(b - a, b_low - a)
        elif face == 3:
            axis = wp.cross(c - b, c_low - b)
        elif face == 4:
            axis = wp.cross(a - c, a_low - c)
        if not apart:
            apart, depth, normal = _hull_overlap_along(
                axis,
                hull_vertices,
                vertex_span,
                a,
                b,
                c,
                a_low,
                b_low,
                c_low,
                depth,
                normal,
            )
    for index in range(normal_span[0], normal_span[0] + normal_span[1]):
        if not apart:
            apart, depth, normal = _hull_overlap_along(
                hull_normals[index],
                hull_vertices,
                vertex_span,
                a,
                b,
                c,
                a_low,
                b_low,
                c_low,
                depth,
                normal,
            )
    for index in range(edge_span[0], edge_span[0] + edge_span[1]):
        for column_edge in range(7):
            edge = b - a
            if column_edge == 1:
                edge = c - b
            elif column_edge == 2:
                edge = a - c
            elif column_edge == 3:
                edge = b_low - a_low
            elif column_edge == 4:
                edge = c_low - b_low
            elif column_edge == 5:
                edge = a_low - c_low
            elif column_edge == 6:
                edge = a - a_low
            if not apart:
                apart, depth, normal = _hull_overlap_along(
                    wp.cross(hull_edges[index], edge),
                    hull_vertices,
                    vertex_span,
                    a,
                    b,
                    c,
                    a_low,
                    b_low,
                    c_low,
                    depth,
                    normal,
                )
    if apart:
        return float(FAR), wp.vec3(0.0, 0.0, 1.0), centre, centre

    # The deepest features along the push: the hull's lowest vertices and
    # the column's highest corners, within 0.1 um.
    hull_min = float(FAR)  # noqa: UP018 (Warp: a variable the loop changes)
    for index in range(vertex_span[0], vertex_span[0] + vertex_span[1]):
        hull_min = wp.min(hull_min, wp.dot(normal, hull_vertices[index]))
    column_max = float(-FAR)  # noqa: UP018 (Warp: a variable the loop changes)
    for corner in range(6):
        column_max = wp.max(
            column_max,
            wp.dot(normal, _column_corner(corner, a, b, c, a_low, b_low, c_low)),
        )
    depth = column_max - hull_min
    tolerance = 1.0e-7
    hull_count = int(0)  # noqa: UP018 (Warp: a variable the loop changes)
    hull_sum = wp.vec3(0.0, 0.0, 0.0)
    hull_first = wp.vec3(0.0, 0.0, 0.0)
    hull_second = wp.vec3(0.0, 0.0, 0.0)
    for index in range(vertex_span[0], vertex_span[0] + vertex_span[1]):
        vertex = hull_vertices[index]
        if wp.dot(normal, vertex) <= hull_min + tolerance:
            if hull_count == 0:
                hull_first = vertex
            elif hull_count == 1:
                hull_second = vertex
            hull_sum = hull_sum + vertex
            hull_count += 1
    column_count = int(0)  # noqa: UP018 (Warp: a variable the loop changes)
    column_sum = wp.vec3(0.0, 0.0, 0.0)
    column_first = wp.vec3(0.0, 0.0, 0.0)
    column_second = wp.vec3(0.0, 0.0, 0.0)
    for corner in range(6):
        vertex = _column_corner(corner, a, b, c, a_low, b_low, c_low)
        if wp.dot(normal, vertex) >= column_max - tolerance:
            if column_count == 0:
                column_first = vertex
            elif column_count == 1:
                column_second = vertex
            column_sum = column_sum + vertex
            column_count += 1
    hull_point = hull_first
    column_point = hull_first + depth * normal
    if hull_count == 1:
        column_point = hull_first + depth * normal
    elif column_count == 1:
        column_point = column_first
        hull_point = column_first - depth * normal
    elif hull_count == 2 and column_count == 2:
        hull_point, column_point = _closest_between_segments(
            hull_first, hull_second, column_first, column_second
        )
    elif column_count <= hull_count:
        column_point = column_sum / float(column_count)
        hull_point = column_point - depth * normal
    else:
        hull_point = hull_sum / float(hull_count)
        column_point = hull_point + depth * normal
    return (
        -depth,
        rotation @ normal,
        rotation @ hull_point + centre,
        rotation @ column_point + centre,
    )


@wp.func
def _write_contact(
    naconmax: int,
    world_index: int,
    distance: float,
    position: wp.vec3,
    normal: wp.vec3,
    margin: float,
    condim: int,
    friction: vec5,
    solref: wp.vec2,
    solreffriction: wp.vec2,
    solimp: vec5,
    adhesion: float,
    geoms: wp.vec2i,
    contact_dist: wp.array1d[float],
    contact_pos: wp.array1d[wp.vec3],
    contact_frame: wp.array1d[wp.mat33],
    contact_includemargin: wp.array1d[float],
    contact_friction: wp.array1d[vec5],
    contact_solref: wp.array1d[wp.vec2],
    contact_solreffriction: wp.array1d[wp.vec2],
    contact_solimp: wp.array1d[vec5],
    contact_dim: wp.array1d[int],
    contact_geom: wp.array1d[wp.vec2i],
    contact_efc_address: wp.array2d[int],
    contact_worldid: wp.array1d[int],
    contact_type: wp.array1d[int],
    contact_geomcollisionid: wp.array1d[int],
    contact_adhesion: wp.array1d[float],
    nacon: wp.array1d[int],
    overflow: wp.array1d[int],
):
    """Append one contact to MuJoCo Warp's pool, as its own ``write_contact``
    does for a pair without an explicit pair or a collision sensor; a full
    pool is an overflow, which stops the run."""
    contact_index = wp.atomic_add(nacon, 0, 1)
    if contact_index >= naconmax:
        wp.atomic_or(
            overflow, world_index, wp.static(int(mjw.OverflowType.NARROWPHASE))
        )
        return
    contact_dist[contact_index] = distance
    contact_pos[contact_index] = position
    contact_frame[contact_index] = make_frame(normal)
    contact_geom[contact_index] = geoms
    contact_worldid[contact_index] = world_index
    contact_includemargin[contact_index] = margin
    contact_dim[contact_index] = wp.where(
        adhesion != 0.0 and distance >= margin, 1, condim
    )
    contact_friction[contact_index] = friction
    contact_solref[contact_index] = solref
    contact_solreffriction[contact_index] = solreffriction
    contact_solimp[contact_index] = solimp
    contact_adhesion[contact_index] = adhesion
    contact_type[contact_index] = wp.static(int(ContactType.CONSTRAINT))
    contact_geomcollisionid[contact_index] = -1
    for row in range(contact_efc_address.shape[1]):
        contact_efc_address[contact_index, row] = -1


@wp.kernel
def _add_ground_contacts(
    # Model:
    hfield_size: wp.array1d[wp.vec4],
    hfield_nrow: wp.array1d[int],
    hfield_ncol: wp.array1d[int],
    hfield_adr: wp.array1d[int],
    hfield_data: wp.array1d[float],
    # The pairs, a height field and a sphere, capsule or convex mesh each:
    pair_geoms: wp.array1d[wp.vec2i],
    pair_field: wp.array1d[int],
    pair_kind: wp.array1d[int],
    pair_hull_vertices: wp.array1d[wp.vec2i],
    pair_hull_normals: wp.array1d[wp.vec2i],
    pair_hull_edges: wp.array1d[wp.vec2i],
    hull_vertices: wp.array1d[wp.vec3],
    hull_normals: wp.array1d[wp.vec3],
    hull_edges: wp.array1d[wp.vec3],
    pair_radius: wp.array1d[float],
    pair_half_length: wp.array1d[float],
    pair_margin: wp.array1d[float],
    pair_gap: wp.array1d[float],
    pair_condim: wp.array1d[int],
    pair_friction: wp.array1d[vec5],
    pair_solref: wp.array1d[wp.vec2],
    pair_solreffriction: wp.array1d[wp.vec2],
    pair_solimp: wp.array1d[vec5],
    pair_adhesion: wp.array1d[float],
    # Data in:
    naconmax: int,
    geom_xpos: wp.array2d[wp.vec3],
    geom_xmat: wp.array2d[wp.mat33],
    # Data out:
    contact_dist: wp.array1d[float],
    contact_pos: wp.array1d[wp.vec3],
    contact_frame: wp.array1d[wp.mat33],
    contact_includemargin: wp.array1d[float],
    contact_friction: wp.array1d[vec5],
    contact_solref: wp.array1d[wp.vec2],
    contact_solreffriction: wp.array1d[wp.vec2],
    contact_solimp: wp.array1d[vec5],
    contact_dim: wp.array1d[int],
    contact_geom: wp.array1d[wp.vec2i],
    contact_efc_address: wp.array2d[int],
    contact_worldid: wp.array1d[int],
    contact_type: wp.array1d[int],
    contact_geomcollisionid: wp.array1d[int],
    contact_adhesion: wp.array1d[float],
    nacon: wp.array1d[int],
    overflow: wp.array1d[int],
):
    """One world's contacts between one height field and one sphere,
    capsule or convex mesh: one for every column the shape overlaps
    (MuJoCo's ``mjc_ConvexHField``), each at the shortest way out of that
    column."""
    world_index, pair_index = wp.tid()
    geoms = pair_geoms[pair_index]
    field = pair_field[pair_index]
    size = hfield_size[field]
    rows = hfield_nrow[field]
    columns = hfield_ncol[field]
    address = hfield_adr[field]
    radius = pair_radius[pair_index]
    margin = pair_margin[pair_index]

    # The shape in the field's frame: a sphere's centre, a capsule's axis or
    # a mesh's convex hull, and the box around it.
    field_position = geom_xpos[world_index, geoms[0]]
    field_rotation = geom_xmat[world_index, geoms[0]]
    to_field = wp.transpose(field_rotation)
    centre = to_field @ (geom_xpos[world_index, geoms[1]] - field_position)
    shape_axes = to_field @ geom_xmat[world_index, geoms[1]]
    kind = pair_kind[pair_index]
    axis = wp.vec3(shape_axes[0, 2], shape_axes[1, 2], shape_axes[2, 2])
    half = pair_half_length[pair_index] * axis
    s0 = centre - half
    s1 = centre + half
    reach = wp.vec3(radius + margin, radius + margin, radius + margin)
    low = wp.min(s0, s1) - reach
    high = wp.max(s0, s1) + reach
    vertex_span = pair_hull_vertices[pair_index]
    if kind == HULL:
        for index in range(vertex_span[0], vertex_span[0] + vertex_span[1]):
            vertex = shape_axes @ hull_vertices[index] + centre
            low = wp.min(low, vertex - reach)
            high = wp.max(high, vertex + reach)
    if (
        low[0] > size[0]
        or high[0] < -size[0]
        or low[1] > size[1]
        or high[1] < -size[1]
        or low[2] > size[2]
        or high[2] < -size[3]
    ):
        return

    # The columns under the shape, as MuJoCo picks them.
    first_column = wp.max(
        0, int(wp.floor((low[0] + size[0]) / (2.0 * size[0]) * float(columns - 1)))
    )
    last_column = wp.min(
        columns - 1,
        int(wp.ceil((high[0] + size[0]) / (2.0 * size[0]) * float(columns - 1))),
    )
    first_row = wp.max(
        0, int(wp.floor((low[1] + size[1]) / (2.0 * size[1]) * float(rows - 1)))
    )
    last_row = wp.min(
        rows - 1, int(wp.ceil((high[1] + size[1]) / (2.0 * size[1]) * float(rows - 1)))
    )
    step_x = wp.float64(2.0) * wp.float64(size[0]) / wp.float64(columns - 1)
    step_y = wp.float64(2.0) * wp.float64(size[1]) / wp.float64(rows - 1)
    dx = float(step_x)
    dy = float(step_y)
    base = -size[3]

    count = int(0)  # noqa: UP018 (Warp: a variable the branches change)
    for row in range(first_row, last_row):
        for column in range(first_column + 1, last_column + 1):
            # Every length from the cell's corner (column - 1, row), the
            # corner's place found in double precision.
            corner_x = step_x * wp.float64(column - 1) - wp.float64(size[0])
            corner_y = step_y * wp.float64(row) - wp.float64(size[1])
            local0 = wp.vec3(
                float(wp.float64(s0[0]) - corner_x),
                float(wp.float64(s0[1]) - corner_y),
                s0[2],
            )
            local1 = wp.vec3(
                float(wp.float64(s1[0]) - corner_x),
                float(wp.float64(s1[1]) - corner_y),
                s1[2],
            )
            near_low = wp.vec3(
                0.0, 0.0, hfield_data[address + row * columns + column - 1] * size[2]
            )
            near_high = wp.vec3(
                0.0,
                dy,
                hfield_data[address + (row + 1) * columns + column - 1] * size[2],
            )
            far_low = wp.vec3(
                dx, 0.0, hfield_data[address + row * columns + column] * size[2]
            )
            far_high = wp.vec3(
                dx, dy, hfield_data[address + (row + 1) * columns + column] * size[2]
            )
            for triangle in range(2):
                # MuJoCo's two triangles of a cell.
                a = near_high
                b = near_low
                c = far_high
                if triangle == 1:
                    a = near_low
                    b = far_high
                    c = far_low
                # A column whose top is all below the shape is skipped.
                distance = float(FAR)  # noqa: UP018 (Warp: a variable the branches change)
                out = wp.vec3(0.0, 0.0, 1.0)
                shape_point = local0
                column_point = local0
                if a[2] >= low[2] or b[2] >= low[2] or c[2] >= low[2]:
                    if kind == HULL:
                        distance, out, shape_point, column_point = _hull_and_column(
                            local0,
                            shape_axes,
                            a,
                            b,
                            c,
                            base,
                            hull_vertices,
                            vertex_span,
                            hull_normals,
                            pair_hull_normals[pair_index],
                            hull_edges,
                            pair_hull_edges[pair_index],
                        )
                    elif kind == CAPSULE:
                        distance, out, shape_point, column_point = _segment_and_column(
                            local0, local1, a, b, c, base
                        )
                    else:
                        distance, out, column_point = _point_and_column(
                            local0, a, b, c, base
                        )
                contact_distance = distance - radius
                if contact_distance < margin + pair_gap[pair_index]:
                    surface_point = shape_point - radius * out
                    middle = 0.5 * (column_point + surface_point)
                    position = wp.vec3(
                        float(wp.float64(middle[0]) + corner_x),
                        float(wp.float64(middle[1]) + corner_y),
                        middle[2],
                    )
                    _write_contact(
                        naconmax,
                        world_index,
                        contact_distance,
                        field_rotation @ position + field_position,
                        field_rotation @ out,
                        margin,
                        pair_condim[pair_index],
                        pair_friction[pair_index],
                        pair_solref[pair_index],
                        pair_solreffriction[pair_index],
                        pair_solimp[pair_index],
                        pair_adhesion[pair_index],
                        geoms,
                        contact_dist,
                        contact_pos,
                        contact_frame,
                        contact_includemargin,
                        contact_friction,
                        contact_solref,
                        contact_solreffriction,
                        contact_solimp,
                        contact_dim,
                        contact_geom,
                        contact_efc_address,
                        contact_worldid,
                        contact_type,
                        contact_geomcollisionid,
                        contact_adhesion,
                        nacon,
                        overflow,
                    )
                    count += 1
                    if count >= MAX_CONTACTS_PER_SHAPE:
                        return


@wp.kernel
def _pair_parameters(
    # Model:
    geom_condim: wp.array1d[int],
    geom_priority: wp.array1d[int],
    geom_solmix: wp.array2d[float],
    geom_solref: wp.array2d[wp.vec2],
    geom_solimp: wp.array2d[vec5],
    geom_friction: wp.array2d[wp.vec3],
    geom_margin: wp.array2d[float],
    geom_gap: wp.array2d[float],
    geom_adhesion: wp.array2d[float],
    pair_dim: wp.array1d[int],
    pair_solref: wp.array2d[wp.vec2],
    pair_solreffriction: wp.array2d[wp.vec2],
    pair_solimp: wp.array2d[vec5],
    pair_margin: wp.array2d[float],
    pair_gap: wp.array2d[float],
    pair_adhesion: wp.array2d[float],
    pair_friction: wp.array2d[vec5],
    # In:
    pairs: wp.array1d[wp.vec2i],
    pair_ids: wp.array1d[wp.vec2i],
    # Out:
    margin_out: wp.array1d[float],
    gap_out: wp.array1d[float],
    condim_out: wp.array1d[int],
    friction_out: wp.array1d[vec5],
    solref_out: wp.array1d[wp.vec2],
    solreffriction_out: wp.array1d[wp.vec2],
    solimp_out: wp.array1d[vec5],
    adhesion_out: wp.array1d[float],
):
    """Each pair's contact parameters, mixed by MuJoCo Warp's own rule."""
    index = wp.tid()
    _, margin, gap, condim, friction, solref, solreffriction, solimp, adhesion = (
        contact_params(
            geom_condim,
            geom_priority,
            geom_solmix,
            geom_solref,
            geom_solimp,
            geom_friction,
            geom_margin,
            geom_gap,
            geom_adhesion,
            pair_dim,
            pair_solref,
            pair_solreffriction,
            pair_solimp,
            pair_margin,
            pair_gap,
            pair_adhesion,
            pair_friction,
            pairs,
            pair_ids,
            index,
            0,
        )
    )
    margin_out[index] = margin
    gap_out[index] = gap
    condim_out[index] = condim
    friction_out[index] = friction
    solref_out[index] = solref
    solreffriction_out[index] = solreffriction
    solimp_out[index] = solimp
    adhesion_out[index] = adhesion


def convex_hull(
    model: mujoco.MjModel, mesh: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A mesh's convex hull, the shape MuJoCo collides, in the mesh's frame:
    its vertices, its distinct face normals and its distinct edge
    directions, from MuJoCo's own hull (``mesh_graph``)."""
    address = model.mesh_graphadr[mesh]
    if address < 0:
        raise ValueError(f"Mesh {model.mesh(mesh).name} has no convex hull.")
    graph = model.mesh_graph[address:]
    vertex_count, face_count = int(graph[0]), int(graph[1])
    first = model.mesh_vertadr[mesh]
    ids = graph[2 + vertex_count : 2 + 2 * vertex_count]
    faces = graph[
        2 + 3 * vertex_count + 3 * face_count : 2 + 3 * vertex_count + 6 * face_count
    ].reshape(-1, 3)
    corners = model.mesh_vert[first + faces]  # (F, 3 corners, 3)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    edges = np.concatenate([corners[:, (i + 1) % 3] - corners[:, i] for i in range(3)])
    edges /= np.linalg.norm(edges, axis=1, keepdims=True)

    def distinct(directions: np.ndarray, either_way: bool) -> np.ndarray:
        kept: list[np.ndarray] = []
        for direction in directions:
            alike = [
                abs(np.dot(direction, k)) if either_way else np.dot(direction, k)
                for k in kept
            ]
            if not alike or max(alike) < 1 - 1e-9:
                kept.append(direction)
        return np.array(kept)

    # The separating-axis test needs each direction once; a face normal's
    # sign does not matter to it either.
    return (
        model.mesh_vert[first + ids],
        distinct(normals, either_way=True),
        distinct(edges, either_way=True),
    )


class GroundContacts:
    """The contacts of a model's spheres, capsules and convex meshes with its
    height fields, made on the GPU by MuJoCo's rule (see the module's
    description).

    ``find`` looks for them in the host model; MuJoCo Warp is built from
    ``gpu_source``, a copy in which those pairs never meet; ``attach`` takes
    that GPU model; ``add`` appends the contacts of a GPU data's positions to
    its contact pool, after MuJoCo Warp's collision stage.
    """

    @classmethod
    def find(cls, model: mujoco.MjModel) -> "GroundContacts | None":
        """The model's ground contacts, or None if it has no height field."""
        fields = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD)
        if len(fields) == 0:
            return None
        return cls(model, fields)

    def __init__(self, model: mujoco.MjModel, fields: np.ndarray) -> None:
        explicit = np.concatenate((model.pair_geom1, model.pair_geom2))
        if np.isin(explicit, fields).any():
            raise ValueError(
                "An explicit contact pair names a height field, which"
                " GroundContacts cannot make: give it no <pair>."
            )

        def meets(first: int, second: int) -> bool:
            return bool(
                model.geom_contype[first] & model.geom_conaffinity[second]
                or model.geom_contype[second] & model.geom_conaffinity[first]
            )

        moving = np.flatnonzero(model.geom_bodyid != 0)
        touching = [g for g in moving if any(meets(f, g) for f in fields)]
        kinds = {
            mujoco.mjtGeom.mjGEOM_SPHERE: 0,
            mujoco.mjtGeom.mjGEOM_CAPSULE: 1,
            mujoco.mjtGeom.mjGEOM_MESH: 2,
        }
        shapes = [g for g in touching if model.geom_type[g] in kinds]
        # Other shapes, if any, keep MuJoCo Warp's contacts.
        others = [g for g in touching if model.geom_type[g] not in kinds]
        # A collision bit no geom uses: the fields keep meeting only the
        # other shapes, through it.
        used = np.bitwise_or.reduce(
            np.concatenate((model.geom_contype, model.geom_conaffinity))
        )
        free_bit = next(1 << bit for bit in range(31) if not used & (1 << bit))
        self.gpu_source = copy.deepcopy(model)
        self.gpu_source.geom_contype[fields] = free_bit
        self.gpu_source.geom_conaffinity[fields] = 0
        self.gpu_source.geom_conaffinity[others] |= free_bit

        pairs = [(int(f), int(s)) for f in fields for s in shapes]
        self.pair_count = len(pairs)
        self._pairs = np.array(pairs, dtype=np.int32).reshape(-1, 2)
        self._field_ids = np.array(
            [model.geom_dataid[f] for f, _ in pairs], dtype=np.int32
        )
        self._radii = np.array([model.geom_size[s][0] for _, s in pairs], np.float32)
        self._kinds = np.array(
            [kinds[model.geom_type[s]] for _, s in pairs], dtype=np.int32
        )
        # A sphere's or a mesh's core is a point: its half length is 0, and a
        # mesh's radius too.
        self._radii[self._kinds == 2] = 0.0
        self._half_lengths = np.array(
            [
                model.geom_size[s][1] if kind == 1 else 0.0
                for (_, s), kind in zip(pairs, self._kinds, strict=True)
            ],
            np.float32,
        )
        # Each mesh's hull once, and each pair's spans of it: first and count.
        hulls: dict[int, tuple[int, int, int]] = {}
        tables: list[list[np.ndarray]] = [[], [], []]
        self._hull_spans = np.zeros((len(pairs), 3, 2), dtype=np.int32)
        for index, ((_, shape), kind) in enumerate(
            zip(pairs, self._kinds, strict=True)
        ):
            if kind != 2:
                continue
            mesh = int(model.geom_dataid[shape])
            if mesh not in hulls:
                hulls[mesh] = tuple(
                    sum(len(part) for part in table) for table in tables
                )
                for table, part in zip(tables, convex_hull(model, mesh), strict=True):
                    table.append(part)
            for which in range(3):
                self._hull_spans[index, which] = (
                    hulls[mesh][which],
                    len(tables[which][list(hulls).index(mesh)]),
                )
        self._hull_tables = [
            np.concatenate(table).astype(np.float32)
            if table
            else np.zeros((1, 3), np.float32)
            for table in tables
        ]

    def attach(self, gpu_model) -> None:
        """Copy the pairs to the GPU, with their contact parameters mixed by
        MuJoCo Warp from the GPU model built from ``gpu_source``."""
        device = gpu_model.geom_type.device
        count = self.pair_count
        self.pair_geoms = wp.array(self._pairs, dtype=wp.vec2i, device=device)
        self.pair_field = wp.array(self._field_ids, dtype=int, device=device)
        self.pair_radius = wp.array(self._radii, dtype=float, device=device)
        self.pair_half_length = wp.array(self._half_lengths, dtype=float, device=device)
        self.pair_kind = wp.array(self._kinds, dtype=int, device=device)
        self.pair_hull_vertices, self.pair_hull_normals, self.pair_hull_edges = (
            wp.array(
                np.ascontiguousarray(self._hull_spans[:, which]),
                dtype=wp.vec2i,
                device=device,
            )
            for which in range(3)
        )
        self.hull_vertices, self.hull_normals, self.hull_edges = (
            wp.array(table, dtype=wp.vec3, device=device) for table in self._hull_tables
        )
        self.pair_margin = wp.zeros(count, dtype=float, device=device)
        self.pair_gap = wp.zeros(count, dtype=float, device=device)
        self.pair_condim = wp.zeros(count, dtype=int, device=device)
        self.pair_friction = wp.zeros(count, dtype=vec5, device=device)
        self.pair_solref = wp.zeros(count, dtype=wp.vec2, device=device)
        self.pair_solreffriction = wp.zeros(count, dtype=wp.vec2, device=device)
        self.pair_solimp = wp.zeros(count, dtype=vec5, device=device)
        self.pair_adhesion = wp.zeros(count, dtype=float, device=device)
        # No explicit pair and no collision sensor.
        no_ids = wp.array(
            np.full((count, 2), -1, dtype=np.int32), dtype=wp.vec2i, device=device
        )
        m = gpu_model
        wp.launch(
            _pair_parameters,
            dim=count,
            inputs=[
                m.geom_condim,
                m.geom_priority,
                m.geom_solmix,
                m.geom_solref,
                m.geom_solimp,
                m.geom_friction,
                m.geom_margin,
                m.geom_gap,
                m.geom_adhesion,
                m.pair_dim,
                m.pair_solref,
                m.pair_solreffriction,
                m.pair_solimp,
                m.pair_margin,
                m.pair_gap,
                m.pair_adhesion,
                m.pair_friction,
                self.pair_geoms,
                no_ids,
            ],
            outputs=[
                self.pair_margin,
                self.pair_gap,
                self.pair_condim,
                self.pair_friction,
                self.pair_solref,
                self.pair_solreffriction,
                self.pair_solimp,
                self.pair_adhesion,
            ],
            device=device,
        )

    def add(self, gpu_model, gpu_data) -> None:
        """Append the contacts of the data's shape positions (after
        ``mjw.kinematics``) to its contact pool (after ``mjw.collision``)."""
        if self.pair_count == 0:
            return
        m, d = gpu_model, gpu_data
        contact = d.contact
        wp.launch(
            _add_ground_contacts,
            dim=(d.nworld, self.pair_count),
            inputs=[
                m.hfield_size,
                m.hfield_nrow,
                m.hfield_ncol,
                m.hfield_adr,
                m.hfield_data,
                self.pair_geoms,
                self.pair_field,
                self.pair_kind,
                self.pair_hull_vertices,
                self.pair_hull_normals,
                self.pair_hull_edges,
                self.hull_vertices,
                self.hull_normals,
                self.hull_edges,
                self.pair_radius,
                self.pair_half_length,
                self.pair_margin,
                self.pair_gap,
                self.pair_condim,
                self.pair_friction,
                self.pair_solref,
                self.pair_solreffriction,
                self.pair_solimp,
                self.pair_adhesion,
                d.naconmax,
                d.geom_xpos,
                d.geom_xmat,
            ],
            outputs=[
                contact.dist,
                contact.pos,
                contact.frame,
                contact.includemargin,
                contact.friction,
                contact.solref,
                contact.solreffriction,
                contact.solimp,
                contact.dim,
                contact.geom,
                contact.efc_address,
                contact.worldid,
                contact.type,
                contact.geomcollisionid,
                contact.adhesion,
                d.nacon,
                d.overflow,
            ],
            device=m.geom_type.device,
        )
