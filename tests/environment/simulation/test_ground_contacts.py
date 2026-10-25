"""Tests for the GPU's contacts of the body with a height field."""

import copy

import mujoco
import numpy as np
import pytest
import torch

mjw = pytest.importorskip("mujoco_warp")
if not torch.cuda.is_available():
    pytest.skip("The ground contacts need a CUDA GPU", allow_module_level=True)
# A physics check (pyproject.toml): run with -m physics when the simulation changes.
pytestmark = pytest.mark.physics

import warp as wp  # noqa: E402

from centipede.environment.simulation.ground_contacts import (  # noqa: E402
    GroundContacts,
    _point_and_column,
    convex_hull,
)
from centipede.environment.simulation.model_mapping import ModelMapping  # noqa: E402
from centipede.environment.simulation.settings import (  # noqa: E402
    SimulationSettings,
)
from centipede.environment.simulation.simulation import model_spec  # noqa: E402

# Stones 3 mm apart, up to 6 mm high: slopes up to 63 degrees between
# neighbours, as on the park's hard grounds, 4 m from the origin as the
# park's far grounds are.
CELLS, CELL_M, RELIEF_M, PLACE_M = 61, 0.003, 0.006, 4.0
# The GPU holds positions 4 m from the origin in float32, about 0.5 um apart.
DEPTH_TOLERANCE_M = 2e-6


def converged(model):
    """MuJoCo's CPU check run to convergence, the reference."""
    model.opt.ccd_iterations = 1000
    model.opt.ccd_tolerance = 1e-10
    return model


def rough_model(folder):
    heights = np.random.default_rng(0).uniform(0, 1, (CELLS, CELLS))
    (folder / "rough.hfield").write_bytes(
        np.array(heights.shape, np.int32).tobytes()
        + heights.astype(np.float32).tobytes()
    )
    half = (CELLS - 1) * CELL_M / 2
    (folder / "rough.xml").write_text(
        f"""<mujoco>
  <asset>
    <hfield name="ground" file="rough.hfield" size="{half} {half} {RELIEF_M} 0.01" />
  </asset>
  <worldbody>
    <geom type="hfield" hfield="ground" pos="{PLACE_M} {-PLACE_M} 0" />
  </worldbody>
</mujoco>""",
        encoding="utf-8",
    )
    settings = SimulationSettings.from_section(
        {
            "model_path": "models/assembly_v6_headless.xml",
            "backend": "gpu",
            "terrain_path": str(folder / "rough.xml"),
        }
    )
    return model_spec(settings).compile()


def pressed_poses(model):
    """The body dropped on the stones with its legs thrashing, and each pose
    pressed 0.4 and 0.8 mm deeper, so that some feet's centres are inside
    a column."""
    data = mujoco.MjData(model)
    root = ModelMapping.from_model(model).root_qpos_address
    data.qpos[root : root + 3] += [PLACE_M, -PLACE_M, RELIEF_M]
    generator = np.random.default_rng(1)
    poses = []
    for step in range(300):
        if step % 15 == 0:
            data.ctrl[:] = generator.uniform(-1, 1, model.nu)
        mujoco.mj_step(model, data)
        if step >= 100 and step % 10 == 0:
            for press in (0.0, 0.0004, 0.0008):
                pose = data.qpos.copy()
                pose[root + 2] -= press
                poses.append(pose)
    return np.array(poses)


def collide_on_gpu(model, poses):
    """MuJoCo Warp's contacts of each pose, then GroundContacts': how many
    MuJoCo Warp made, and each contact's world, geoms, depth, normal,
    position and parameters (condim, friction, solref, solimp)."""
    contacts_of = GroundContacts.find(model)
    gpu_model = mjw.put_model(contacts_of.gpu_source)
    contacts_of.attach(gpu_model)
    gpu_data = mjw.make_data(model, nworld=len(poses), nconmax=512, njmax=64)
    gpu_data.qpos.assign(wp.array(poses.astype(np.float32), dtype=float))
    mjw.kinematics(gpu_model, gpu_data)
    mjw.collision(gpu_model, gpu_data)
    made_by_mujoco_warp = int(gpu_data.nacon.numpy()[0])
    contacts_of.add(gpu_model, gpu_data)
    n = int(gpu_data.nacon.numpy()[0])
    c = gpu_data.contact
    params = np.concatenate(
        (
            c.dim.numpy()[:n, None],
            c.friction.numpy()[:n],
            c.solref.numpy()[:n],
            c.solimp.numpy()[:n],
        ),
        axis=1,
    )
    return made_by_mujoco_warp, (
        c.worldid.numpy()[:n],
        c.geom.numpy()[:n],
        c.dist.numpy()[:n],
        c.frame.numpy()[:n, 0],
        c.pos.numpy()[:n],
        params,
    )


def ground_contacts_of(model, ground, contacts):
    """Each foot's and leg link's contacts with the ground, by shape: depth,
    the normal out of the ground, position and parameters."""
    found = {}
    for geom, dist, normal, pos, params in contacts:
        if ground not in geom:
            continue
        shape = int(geom[1] if geom[0] == ground else geom[0])
        if model.geom_type[shape] != mujoco.mjtGeom.mjGEOM_MESH:
            found.setdefault(shape, []).append((dist, normal, pos, params))
    return found


def test_gpu_feet_and_links_touch_the_ground_as_mujoco_on_the_cpu(tmp_path):
    """On steep stones far from the origin, where MuJoCo Warp's own convex
    check put feet millimetres to metres deep, every foot's and leg link's
    contact with the ground matches MuJoCo's on the CPU: the same contacts,
    their depths within 2 um, their normals within 0.5 degrees, their
    parameters equal; and MuJoCo Warp itself makes no contact with the
    ground."""
    model = rough_model(tmp_path)
    ground = int(np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD)[0])
    poses = pressed_poses(model)

    cpu = []
    reference = converged(copy.deepcopy(model))
    data = mujoco.MjData(reference)
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_kinematics(reference, data)
        mujoco.mj_collision(reference, data)
        contacts = [data.contact[i] for i in range(data.ncon)]
        cpu.append(
            ground_contacts_of(
                model,
                ground,
                [
                    (
                        c.geom,
                        c.dist,
                        np.array(c.frame[:3]),
                        np.array(c.pos),
                        (c.dim, *c.friction, *c.solref, *c.solimp),
                    )
                    for c in contacts
                ],
            )
        )
    made_by_mujoco_warp, (world, geom, dist, normal, pos, params) = collide_on_gpu(
        model, poses
    )
    gpu = [
        ground_contacts_of(
            model,
            ground,
            [
                (geom[i], dist[i], normal[i], pos[i], params[i])
                for i in np.flatnonzero(world == index)
            ],
        )
        for index in range(len(poses))
    ]

    assert not (geom[:made_by_mujoco_warp] == ground).any()
    # Some feet are pressed deeper than their radius: centres inside a column.
    spheres = model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE
    feet_depths = [
        depth
        for shapes in cpu
        for shape, found in shapes.items()
        if spheres[shape]
        for depth, *_ in found
    ]
    assert min(feet_depths) < -model.geom_size[spheres, 0].max()

    def matched(theirs, mine):
        """Every contact deeper than 2 um has its match: the same depth
        within 2 um, normal within 0.5 degrees, parameters, and column (a
        link nearly parallel to the face it leaves by is as deep all along,
        and either backend may place the contact anywhere on it)."""
        for depth, normal, place, parameters in theirs:
            if depth > -2e-6:
                continue
            assert any(
                abs(other[0] - depth) < DEPTH_TOLERANCE_M
                and np.dot(normal, other[1]) > np.cos(np.radians(0.5))
                and np.linalg.norm(other[2] - place) < CELL_M
                and np.allclose(other[3], parameters, rtol=1e-6)
                for other in mine
            )

    for on_cpu, on_gpu in zip(cpu, gpu, strict=True):
        for shape in set(on_cpu) | set(on_gpu):
            matched(on_cpu.get(shape, []), on_gpu.get(shape, []))
            matched(on_gpu.get(shape, []), on_cpu.get(shape, []))


def test_gpu_shell_contacts_match_mujocos_convex_meshes():
    """A body shell pressed into the two columns of one steep cell, 4 m from
    the origin, in random poses: every column MuJoCo finds the shell in, as
    convex meshes, has our contact, a true way out of it (the push along its
    normal clears the column) and never longer than MuJoCo's, within 2 um.
    MuJoCo's own height-field check on the CPU is less exact for meshes, by
    up to tens of um, and even its converged mesh check sometimes stops at a
    longer way out (2026-10-24); near the shortest, ways out of almost equal
    length can point apart by degrees, so the normals are not compared."""
    shell = convex_hull(
        mujoco.MjModel.from_xml_path("models/assembly_v6_headless.xml"), 0
    )[0]
    shell_xml = " ".join(f"{x:.9f}" for x in shell.ravel())
    half, base = CELL_M / 2, 0.01
    field = mujoco.MjModel.from_xml_string(f"""<mujoco>
  <asset>
    <hfield name="cell" nrow="2" ncol="2" elevation="0 1 0.25 0.75"
            size="{half} {half} {RELIEF_M} {base}" />
    <mesh name="shell" vertex="{shell_xml}" />
  </asset>
  <worldbody>
    <geom type="hfield" hfield="cell" pos="{PLACE_M} {-PLACE_M} 0" />
    <body><freejoint /><geom type="mesh" mesh="shell" /></body>
  </worldbody>
</mujoco>""")
    # The cell's heights, 0 to 1, as MuJoCo stores them: row r at y = r cells.
    heights = field.hfield_data.reshape(2, 2)

    def corner(column, row):
        return (
            PLACE_M - half + CELL_M * column,
            -PLACE_M - half + CELL_M * row,
            RELIEF_M * heights[row, column],
        )

    columns = [
        (corner(0, 1), corner(0, 0), corner(1, 1)),
        (corner(0, 0), corner(1, 1), corner(1, 0)),
    ]
    # Each column as a convex mesh about its centre, placed there.
    column_xml, centres = "", []
    for index, top in enumerate(columns):
        corners = np.array([*top, *[(x, y, -base) for x, y, _ in top]])
        centres.append(corners.mean(axis=0))
        vertices = " ".join(f"{x:.9f}" for x in (corners - centres[-1]).ravel())
        column_xml += f'<mesh name="column{index}" vertex="{vertices}" />'
    meshes = converged(
        mujoco.MjModel.from_xml_string(f"""<mujoco>
  <asset>{column_xml}<mesh name="shell" vertex="{shell_xml}" /></asset>
  <worldbody>
    <geom type="mesh" mesh="column0" pos="{" ".join(map(str, centres[0]))}" />
    <geom type="mesh" mesh="column1" pos="{" ".join(map(str, centres[1]))}" />
    <body><freejoint /><geom type="mesh" mesh="shell" /></body>
  </worldbody>
</mujoco>""")
    )

    # Random turns, the shell's lowest point 0 to 0.5 mm below the cell's
    # middle, kept when MuJoCo finds a contact deeper than 2 um.
    generator = np.random.default_rng(2)
    data = mujoco.MjData(meshes)
    poses, expected = [], []
    while len(poses) < 40:
        turn = generator.normal(size=4)
        turn /= np.linalg.norm(turn)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, turn)
        lowest = (shell @ rotation.reshape(3, 3).T)[:, 2].min()
        place = [
            PLACE_M + generator.uniform(-half, half),
            -PLACE_M + generator.uniform(-half, half),
            RELIEF_M / 2 - lowest - generator.uniform(0, 0.0005),
        ]
        data.qpos[:] = [*place, *turn]
        mujoco.mj_kinematics(meshes, data)
        mujoco.mj_collision(meshes, data)
        # MuJoCo may give a pair of meshes several contacts: its deepest.
        deepest = {}
        for i in range(data.ncon):
            column = int(data.contact[i].geom[0])
            if data.contact[i].dist < deepest.get(column, (np.inf,))[0]:
                deepest[column] = (
                    data.contact[i].dist,
                    np.array(data.contact[i].frame[:3]),
                )
        if deepest and min(d for d, _ in deepest.values()) < -2e-6:
            poses.append(data.qpos.copy())
            expected.append(deepest)

    _, (world, _, dist, normal, _, _) = collide_on_gpu(field, np.array(poses))
    corners = [np.array([*top, *[(x, y, -base) for x, y, _ in top]]) for top in columns]
    for index, deepest in enumerate(expected):
        data.qpos[:] = poses[index]
        mujoco.mj_kinematics(meshes, data)
        hull = shell @ data.geom_xmat[2].reshape(3, 3).T + data.geom_xpos[2]
        mine = {}
        for i in np.flatnonzero(world == index):
            # Pushed by its depth along its normal, the shell leaves a
            # column (each contact its own: columns may share the way out).
            pushes = [(c @ normal[i]).max() - (hull @ normal[i]).min() for c in corners]
            column = next(
                k
                for k, push in enumerate(pushes)
                if k not in mine and abs(push + dist[i]) < DEPTH_TOLERANCE_M
            )
            mine[column] = (dist[i], normal[i])
        for column, (depth, _) in deepest.items():
            if depth > -2e-6:
                continue
            assert mine[column][0] >= depth - DEPTH_TOLERANCE_M


@wp.kernel
def _ways_out(
    points: wp.array1d[wp.vec3],
    corners: wp.array2d[wp.vec3],
    distances: wp.array1d[float],
    normals: wp.array1d[wp.vec3],
):
    index = wp.tid()
    distance, normal, _ = _point_and_column(
        points[index], corners[index, 0], corners[index, 1], corners[index, 2], -0.01
    )
    distances[index] = distance
    normals[index] = normal


def test_a_point_on_a_columns_top_gets_the_tops_normal():
    """A foot's centre within a nanometre of a column's top, in or out, gets
    the top's normal: single precision once turned the direction toward a
    nearest point that close into 0/0, and one such contact made a run's
    learning non-finite (run 25, 2026-10-25)."""
    count = 100_000
    rng = np.random.default_rng(0)
    heights = rng.uniform(0, RELIEF_M, (count, 3))
    # MuJoCo's first triangle of a cell, in the cell corner's frame.
    a = np.stack([np.zeros(count), np.full(count, CELL_M), heights[:, 0]], 1)
    b = np.stack([np.zeros(count), np.zeros(count), heights[:, 1]], 1)
    c = np.stack([np.full(count, CELL_M), np.full(count, CELL_M), heights[:, 2]], 1)
    top = np.cross(b - a, c - a)
    top /= np.linalg.norm(top, axis=1, keepdims=True)
    # Well inside the triangle, then nudged along the top's normal.
    u, v = rng.uniform(0.05, 0.45, (2, count))
    points = a + u[:, None] * (b - a) + v[:, None] * (c - a)
    points += rng.uniform(-1e-9, 1e-9, (count, 1)) * top
    distances = wp.zeros(count, dtype=float)
    normals = wp.zeros(count, dtype=wp.vec3)
    wp.launch(
        _ways_out,
        dim=count,
        inputs=[
            wp.array(points.astype(np.float32), dtype=wp.vec3),
            wp.array(np.stack([a, b, c], 1).astype(np.float32), dtype=wp.vec3),
        ],
        outputs=[distances, normals],
    )
    assert np.abs(distances.numpy()).max() < 1e-8
    np.testing.assert_allclose(normals.numpy(), top, atol=1e-5)
