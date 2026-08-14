"""Small helpers shared by the qualification and phase1 policies."""

from aic_control_interfaces.msg import MotionUpdate


def build_motion_update(pos, quat, stiffness, damping):
    """Build a MotionUpdate for a Cartesian target pose with diagonal stiffness/damping."""
    motion_update = MotionUpdate()
    motion_update.header.frame_id = "base_link"
    motion_update.trajectory_generation_mode.mode = 2

    motion_update.pose.position.x = float(pos[0])
    motion_update.pose.position.y = float(pos[1])
    motion_update.pose.position.z = float(pos[2])
    motion_update.pose.orientation.x = float(quat[0])
    motion_update.pose.orientation.y = float(quat[1])
    motion_update.pose.orientation.z = float(quat[2])
    motion_update.pose.orientation.w = float(quat[3])

    mat_stiff = [0.0] * 36
    mat_damp = [0.0] * 36
    for j in range(6):
        mat_stiff[j * 6 + j] = float(stiffness[j])
        mat_damp[j * 6 + j] = float(damping[j])
    motion_update.target_stiffness = mat_stiff
    motion_update.target_damping = mat_damp

    return motion_update


def check_force_threshold(observation, logger, threshold_n=20.0):
    """Log a warning if any wrist force axis exceeds threshold_n."""
    try:
        if hasattr(observation, 'wrist_wrench') and observation.wrist_wrench is not None:
            force = observation.wrist_wrench.wrench.force
            fx, fy, fz = force.x, force.y, force.z

            if abs(fx) > threshold_n or abs(fy) > threshold_n or abs(fz) > threshold_n:
                logger.warning(
                    f"⚠️ HOHE KRAFT! FX: {fx:6.2f} N | FY: {fy:6.2f} N | FZ: {fz:6.2f} N"
                )
                return True
    except Exception:
        pass
    return False
