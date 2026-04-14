"""Hilfsfunktionen fuer TF-Lookups und Quaternion-/Vektor-Transformationen.

Dieses Modul enthaelt bewusst nur kleine, wiederverwendbare Funktionen,
damit die eigentliche Policy-Logik in Plug_in.py uebersichtlich bleibt.
"""

from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.time import Time


def lookup_pose_in_base(tf_buffer, frame_id: str, base_frame: str = "base_link"):
    """Liest die Pose eines Frames relativ zum Basis-Frame aus TF.

    Args:
        tf_buffer: ROS2 TF-Buffer (normalerweise node._tf_buffer).
        frame_id: Ziel-Frame, dessen Pose gelesen werden soll.
        base_frame: Referenz-Frame, standardmaessig "base_link".

    Returns:
        Tuple (translation, rotation) aus geometry_msgs-Typen.
    """
    transform = tf_buffer.lookup_transform(base_frame, frame_id, Time())
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    return translation, rotation


def rotate_vector_by_quaternion(quaternion: Quaternion, vector):
    """Rotiert einen 3D-Vektor mit einer Quaternion.

    Die Rechnung entspricht: v' = q * v * q^-1.
    """
    qx = quaternion.x
    qy = quaternion.y
    qz = quaternion.z
    qw = quaternion.w
    vx, vy, vz = vector

    # q * v (mit v als Quaternion [vx, vy, vz, 0])
    ix = qw * vx + qy * vz - qz * vy
    iy = qw * vy + qz * vx - qx * vz
    iz = qw * vz + qx * vy - qy * vx
    iw = -qx * vx - qy * vy - qz * vz

    # (q * v) * q^-1
    ox = ix * qw + iw * -qx + iy * -qz - iz * -qy
    oy = iy * qw + iw * -qy + iz * -qx - ix * -qz
    oz = iz * qw + iw * -qz + ix * -qy - iy * -qx
    return ox, oy, oz


def quaternion_multiply(q1: Quaternion, q2: Quaternion):
    """Multipliziert zwei Quaternionen und liefert die zusammengesetzte Rotation."""
    return Quaternion(
        x=q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        y=q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        z=q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        w=q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
    )


def compute_tcp_target_pose(target_pos_base_link, desired_tip_rotation: Quaternion, tip_from_tcp):
    """Berechnet die TCP-Zielpose, sodass die Spitze auf der Entrance-Pose sitzt.

    Args:
        target_pos_base_link: Translation des Ziel-Frames in base_link.
        desired_tip_rotation: Gewuenschte Orientierung der Spitze in base_link.
        tip_from_tcp: Transform T_tip_tcp aus TF (lookup tip <- tcp).

    Returns:
        geometry_msgs.msg.Pose fuer den TCP in base_link.
    """
    # Offset tip<-tcp in die Basisorientierung des Ziel-Tips drehen.
    tip_from_tcp_t = tip_from_tcp.transform.translation
    tip_from_tcp_q = tip_from_tcp.transform.rotation
    offset_base_x, offset_base_y, offset_base_z = rotate_vector_by_quaternion(
        desired_tip_rotation,
        (tip_from_tcp_t.x, tip_from_tcp_t.y, tip_from_tcp_t.z),
    )

    # Gesamtrotation fuer den TCP in base_link.
    target_rotation = quaternion_multiply(desired_tip_rotation, tip_from_tcp_q)

    return Pose(
        position=Point(
            x=target_pos_base_link.x + offset_base_x,
            y=target_pos_base_link.y + offset_base_y,
            z=target_pos_base_link.z + offset_base_z,
        ),
        orientation=Quaternion(
            x=target_rotation.x,
            y=target_rotation.y,
            z=target_rotation.z,
            w=target_rotation.w,
        ),
    )
