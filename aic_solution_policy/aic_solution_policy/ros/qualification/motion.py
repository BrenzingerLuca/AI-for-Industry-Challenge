"""Cartesian motion primitives for the qualification-round policy: a smooth
move to a target pose, and a spiral search/insert around a center point.
"""

import math

import numpy as np

from ..common import build_motion_update, check_force_threshold


def move_tcp_smooth_cartesian(pos, quat, move_robot, get_observation,
                               stiffness, damping, sleep_for, logger,
                               n_steps=80, label="Target"):
    """Commands a single target pose and holds it for n_steps, checking forces
    and the remaining distance each step. Returns early once within 1mm.
    """
    motion_update = build_motion_update(pos, quat, stiffness, damping)

    logger.info(f"==> Move to {label} (smooth): P=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    dist = float('inf')
    for i in range(n_steps):
        move_robot(motion_update=motion_update)
        obs = get_observation()
        check_force_threshold(obs, logger)

        curr = obs.controller_state.tcp_pose.position
        dist = math.sqrt(
            (curr.x - pos[0]) ** 2 +
            (curr.y - pos[1]) ** 2 +
            (curr.z - pos[2]) ** 2
        )

        if i % 25 == 0:
            logger.info(f"    [{i}] Distanz zu {label}: {dist * 1000:.2f} mm")
            if dist < 0.001:  # 1mm threshold
                logger.info(f"    Ziel erreicht! Restfehler: {dist * 1000:.3f} mm")
                return dist

        sleep_for(0.1)
    logger.info(f"    [{label}] Fertig. Restfehler: {dist * 1000:.3f} mm")

    return dist


def spiral_search_and_insert(center_pos, quat, move_robot, get_observation,
                              sleep_for, logger,
                              stiff_spiral=[300.0, 300.0, 80.0, 200.0, 200.0, 200.0],
                              damp_spiral=[40.0, 40.0, 15.0, 30.0, 30.0, 30.0],
                              spiral_steps=120, max_radius=0.003, n_turns=3,
                              label="Spiral"):
    """Moves the TCP in an outward spiral around center_pos on a fixed Z plane
    to search for the port opening, using soft contact stiffness so it can
    slip in once it finds the hole.
    """
    logger.info(f"==> Start Spiral search for {label} | max_radius={max_radius * 1000:.1f}mm | turns={n_turns} | steps={spiral_steps}")

    t_vals = np.linspace(0, n_turns * 2 * np.pi, spiral_steps)

    for idx, t in enumerate(t_vals):
        r = (t / (n_turns * 2 * np.pi)) * max_radius
        dx = r * np.cos(t)
        dy = r * np.sin(t)

        search_pos = center_pos.copy()
        search_pos[0] += dx
        search_pos[1] += dy

        # Z stays fixed at center_pos[2].
        motion_update = build_motion_update(search_pos, quat, stiff_spiral, damp_spiral)
        move_robot(motion_update=motion_update)

        obs = get_observation()
        check_force_threshold(obs, logger)

        curr = obs.controller_state.tcp_pose.position
        dist = math.sqrt(
            (curr.x - search_pos[0]) ** 2 +
            (curr.y - search_pos[1]) ** 2 +
            (curr.z - center_pos[2]) ** 2
        )

        if idx % 20 == 0:
            logger.info(f"    [{idx}/{spiral_steps}] r={r * 1000:.2f}mm | dx={dx * 1000:.1f}mm dy={dy * 1000:.1f}mm | dist={dist * 1000:.2f}mm")

        sleep_for(0.05)

    obs = get_observation()
    curr = obs.controller_state.tcp_pose.position
    final_dist = math.sqrt(
        (curr.x - center_pos[0]) ** 2 +
        (curr.y - center_pos[1]) ** 2 +
        (curr.z - center_pos[2]) ** 2
    )

    logger.info(f"    [{label}] Spiral search done, final distance: {final_dist * 1000:.2f}mm")

    return final_dist
