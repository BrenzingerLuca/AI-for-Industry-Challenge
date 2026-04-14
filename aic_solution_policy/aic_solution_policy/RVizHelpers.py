"""Hilfsfunktionen fuer RViz-Visualisierung."""

from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker


def rviz_vector(parent_node, pose: Pose, color="green"):
    """Visualisiert eine Pose als Vektor (Arrow Marker) in RViz.

    Der Marker ist relativ zu `base_link` und wird auf dem Topic
    `vector_marker_base_link` publiziert, damit die Zielausrichtung einfach
    kontrolliert werden kann.

    Args:
        parent_node: ROS2 Node.
        pose: Pose mit Position und Orientierung des Markers in base_link.
        color: Farbe als String ("green", "red", "blue", "yellow", "cyan").
    """
    color_map = {
        "green": (0.1, 0.95, 0.2),
        "red": (0.95, 0.1, 0.2),
        "blue": (0.2, 0.3, 0.95),
        "yellow": (0.95, 0.95, 0.1),
        "cyan": (0.1, 0.95, 0.95),
    }

    # Publisher einmalig lazy anlegen und dann wiederverwenden.
    if not hasattr(parent_node, "_vector_marker_pub"):
        parent_node._vector_marker_pub = parent_node.create_publisher(
            Marker,
            "vector_marker_base_link",
            10,
        )
        parent_node._vector_marker_id_counter = 0

    # Eindeutige ID fuer jeden Marker.
    marker_id = parent_node._vector_marker_id_counter
    parent_node._vector_marker_id_counter = (parent_node._vector_marker_id_counter + 1) % 100

    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = parent_node.get_clock().now().to_msg()
    marker.ns = "vector_marker_base_link"
    marker.id = marker_id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    # Pose steuert Startposition und Orientierung des Vektors.
    marker.pose = pose

    # Arrow-Geometrie: Laenge (x), Dicke (y,z).
    marker.scale.x = 0.08
    marker.scale.y = 0.008
    marker.scale.z = 0.008

    # Farbe aus Map holen und zuweisen (RGB-Tuple + volle Opazitaet).
    r, g, b = color_map.get(color, (0.5, 0.5, 0.5))  # Default grau falls unbekannte Farbe.
    marker.color.r = r
    marker.color.g = g
    marker.color.b = b
    marker.color.a = 1.0

    parent_node._vector_marker_pub.publish(marker)
